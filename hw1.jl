using Pkg
Pkg.activate(".")

using LinearAlgebra 
using SatelliteDynamics 
using Plots 
using Convex
using Clarabel
using BenchmarkTools
#add convex and clarabel to solve the convex semidefinite program

#add quaternion helper functions 
H = [zeros(1,3); I];
T = [1  zeros(1,3);
     zeros(3,1) -I];

function L(q)
    return [q[1]          -q[2:4]';
            q[2:4]    q[1]*I + hat(q[2:4])]
end

function R(q)
    return [q[1]          -q[2:4]';
            q[2:4]    q[1]*I - hat(q[2:4])]
end

function G(q)
    return L(q)*H
end

function Q(q)
    return H'*L(q)*R(q)'*H
end

#hat operator function 
function hat(x)

    x_hat = [0 -x[3] x[2]; x[3] 0 -x[1]; -x[2] x[1] 0]

    return x_hat

end

#unhat operator function 
function unhat(x)
    return 0.5*[x[3,2]-x[2,3];
                x[1,3]-x[3,1];
                x[2,1]-x[1,2]]
end

#center of mass calculation 

#length of each body in their own coordinate system 
#used for moment of inertia calculation 
#see Figure 2

#prism 1 
lx_1 = 10
ly_1 = 20
lz_1 = 34

#prism 2
lx_2 = 1
ly_2 = 100
lz_2 = 34 

#prism 3
lx_3 = 1
ly_3 = 100
lz_3 = 34

#mass of each rectangular prism (in kg)
m1 = 12
m2 = 1
m3 = 1

#distance from the body frame origin to each prism com (in cm) 
r1 = [0,0,0]
r2 = [35.36, 35.36, 0]
r3 = [-35.36, -35.36, 0]

#center of mass
r_com = (m1*r1 + m2*r2 + m3*r3)/(m1 + m2 + m3)

#body frame moment of inertia calculation 

#individual moment of inertias (formula for the rectangular prism)
#source: https://dynref.engr.illinois.edu/rem.html
#Ix = (1/12)*m*(ly^2 + lz^2)
#Iy = (1/12)*m*(lz^2 + lx^2)
#Iz = (1/12)*m*(lx^2 + ly^2)

J_1 = Matrix(1.0*I,3,3).*[(1/12)*m1*(ly_1^2 + lz_1^2); (1/12)*m1*(lz_1^2 + lx_1^2); (1/12)*m1*(lx_1^2 + ly_1^2)]
J_2 = Matrix(1.0*I,3,3).*[(1/12)*m2*(ly_2^2 + lz_2^2); (1/12)*m2*(lz_2^2 + lx_2^2); (1/12)*m2*(lx_2^2 + ly_2^2)]
J_3 = Matrix(1.0*I,3,3).*[(1/12)*m3*(ly_3^2 + lz_3^2); (1/12)*m3*(lz_3^2 + lx_3^2); (1/12)*m3*(lx_3^2 + ly_3^2)]

#combined inertia calculation 
#uses the parallel axis theorem 
J_b = J_1 + J_2 + J_3 + m1*(r1'*r1*I - r1*r1') + m2*(r2'*r2*I - r2*r2') + m3*(r3'*r3*I - r3*r3')
 
#principle axes is the bases where the body frame moment of inertia is diagonal 
#take an eigen decomp to find this 
J_b_eig = eigen(J_b)

#this is a rotation that can convert from principle axes to body frame
principle_axes = J_b_eig.vectors 

J_b_eig.values 
#moment of inertia about principle axes
J_principle = principle_axes'*J_b*principle_axes 

#get into units of meters and seconds
J_principle_SI = J_principle/(100^2)

J_b_SI = J_b/(100^2)

#perturb inertia function 
function perturb_inertia(J)

    #sample random perturbations for eigen values and vectors 
    d = 0.1*randn(3)
    v = 0.1*randn(3)

    J_eig = eigen(J)
    D = J_eig.values.*Matrix(1.0*I,3,3)
    V = J_eig.vectors
    D_tilde = D*(Matrix(1.0*I,3,3) + diagm(d))
    V_tilde = V*exp(hat(v))

    J_tilde = V_tilde*D_tilde*V_tilde'

    return J_tilde 

end
 
#pertub the inertia
J_b_SI_perturbed = perturb_inertia(J_b_SI) 

#desired rpm (rotations per minute)
desired_rpm = 10 

#convert rpm to rad/s 
desired_rad_s = desired_rpm*(2*pi/60)

#solar panel normal 
solar_normal = [-sqrt(2)/2; sqrt(2)/2; 0]

#define continous orbit dynamics function
#state x - [position, velocity], units [km, s]
#current model has gravitation acceleration and J2
function orbit_dynamics(x)

    #position
    r = x[1:3]

    #velocity 
    v = x[4:6]

    #earth gravitational constant [km3/s2]
    μ = GM_EARTH/(1000^3)
    
    #radius of the Earth in km
    r_earth = R_EARTH/1000

    a_grav = (-μ/norm(r)^3)*r 

    a_J2 = (-3/2)*((J2_EARTH*μ*r_earth^2)/(norm(r)^5))*
    [(1-5*(r[3]/norm(r))^2)*r[1]; (1-5*(r[3]/norm(r))^2)*r[2]; (3-5*(r[3]/norm(r))^2)*r[3]]

    #total acceleration
    a = a_grav + a_J2

    x_dot = [v; a]

end

#function to simulate position and attitude 
#x - position, velocity, attitude quaternion, angular velocity, rotor momentum 
function orbit_attitude_dynamics(x, u)

    #position
    r = x[1:3]

    #velocity 
    v = x[4:6]

    #attitude quaternion
    q = x[7:10]

    #angular velocity
    ω = x[11:13]

    #rotor momentum 
    ρ= x[14:16]

    ρ_dot = u 

    #ω_dot = -J_b_SI\(ρ_dot + hat(ω)*(J_b_SI*ω + ρ))

    #with the perturbed inertia 
    ω_dot = -J_b_SI_perturbed\(ρ_dot + hat(ω)*(J_b_SI_perturbed*ω + ρ))

    q_dot = 0.5*G(q)*ω

    #earth gravitational constant [km3/s2]
    μ = GM_EARTH/(1000^3)
    
    #radius of the Earth in km
    r_earth = R_EARTH/1000

    a_grav = (-μ/norm(r)^3)*r 

    a_J2 = (-3/2)*((J2_EARTH*μ*r_earth^2)/(norm(r)^5))*
    [(1-5*(r[3]/norm(r))^2)*r[1]; (1-5*(r[3]/norm(r))^2)*r[2]; (3-5*(r[3]/norm(r))^2)*r[3]]

    #total acceleration
    a = a_grav + a_J2

    x_dot = [v; a ;q_dot; ω_dot; ρ_dot]

    return x_dot

end

# function euler_dynamics_body(x, u)

#     #angular velocity
#     ω = x[1:3]

#     #rotor momentum
#     ρ = x[4:6]

#     ρ_dot = u 

#     ω_dot = -J_b_SI\(ρ_dot + hat(ω)*(J_b_SI*ω + ρ))

#     x_dot = [ω_dot; ρ_dot]

#     return x_dot
# end

#rk4 implementation
#f - dynamics function 
# x - state 
function RK4_integrator(f, x)

    k1 = f(x)
    k2 = f(x + (dt/2)*k1)
    k3 = f(x + (dt/2)*k2)
    k4 = f(x + (dt)*k3)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return xk_1 

end

#assuming zero order hold on controls
function RK4_integrator_wcontrol(f, x, u)

    k1 = f(x, u)
    k2 = f(x + (dt/2)*k1, u)
    k3 = f(x + (dt/2)*k2, u)
    k4 = f(x + (dt)*k3, u)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return xk_1 

end

function RK4_integrator_fullsim(f, x, u)

    k1 = f(x, u)
    k2 = f(x + (dt/2)*k1, u)
    k3 = f(x + (dt/2)*k2, u)
    k4 = f(x + (dt)*k3, u)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    #re normalize the quaternion 
    xk_1[7:10] .= xk_1[7:10]/norm(xk_1[7:10])

    return xk_1 

end



#initial orbit (osculating orbital elements)
oe0  = [R_EARTH + 550e3, 0.0, 90.0, 0, 0, 0]

#period (based on the semimajor axis)
T = orbit_period(oe0[1])

#convert to cartesian (in meters)
eci0 = sOSCtoCART(oe0, use_degrees=true)

#scale to km and km/s 
eci0_km = eci0./1000

#number of discrete points per orbit
N = 100

#integration timestep 
dt = T/(N-1)

#state size 
nx = 6

#number of revs to simulate 
n_revs = 10

#N-1 to not repeat the same starting point every rev 
total_N = (N-1)*n_revs

x_traj = zeros(6, total_N)

x_traj[:,1] = eci0_km

for k=1:total_N-1

    x_traj[:,k+1] = RK4_integrator(orbit_dynamics, x_traj[:,k])

end

plot(x_traj[1,:], x_traj[2,:], x_traj[3,:], title = "Orbit Trajectory", label = "position")
xlabel!("X [km]") 
ylabel!("           Y [km]")
zlabel!("Z [km ]")

#savefig("figures/hw1/orbit_traj.png")

#euler equation dynamics simulation 

#in the principle axis frame
function euler_dynamics(ω)

    ω_dot = zeros(3)

    ω_dot[1] = (-(J_principle_SI[3,3] - J_principle_SI[2,2])*ω[2]*ω[3])/J_principle_SI[1,1] 
    ω_dot[2] = (-(J_principle_SI[1,1] - J_principle_SI[3,3])*ω[1]*ω[3])/J_principle_SI[2,2] 
    ω_dot[3] = (-(J_principle_SI[2,2] - J_principle_SI[1,1])*ω[1]*ω[2])/J_principle_SI[3,3] 

    return ω_dot 

end

#euler equation in the body frame 
#x - [angular velocity; rotor momentum]
#u - [rotor momentum rate]
function euler_dynamics_body(x, u)

    #angular velocity
    ω = x[1:3]

    #rotor momentum
    ρ = x[4:6]

    ρ_dot = u 

    #normal body frame inertia
    #ω_dot = -J_b_SI\(ρ_dot + hat(ω)*(J_b_SI*ω + ρ))

    #perturbed inertia 
    ω_dot = -J_b_SI_perturbed\(ρ_dot + hat(ω)*(J_b_SI_perturbed*ω + ρ))

    x_dot = [ω_dot; ρ_dot]

    return x_dot
end

#timestep attitude sim 
#10 Hz
dt = 0.1

#number of timesteps (~3 minutes)
N = 2000

#state size for omega sim 
nx_ω= 3
ω_traj = zeros(nx_ω,N)

#initial omega 
#start with a magnitude of 10 RPM 

#convert to rad/s
initial_magnitude = 3 * 2*pi/60

#minor axis
#ω0 = [1, 0, 0]*initial_magnitude + 0.1*randn(3)

#intermediate axis
#ω0 = [0, 1, 0]*initial_magnitude + 0.1*randn(3)

#major axis
ω0 = [0, 0, 1]*initial_magnitude + 0.1*randn(3)

ω_traj[:,1] = ω0
#simulate about each principle axis
for k=1:N-1 

    ω_traj[:,k+1] = RK4_integrator(euler_dynamics, ω_traj[:,k])
    
end

ω_traj 

plot(ω_traj[1,:], label="ωx", title="ω Trajectory Minor Axis Spin", linewidth=3)
plot!(ω_traj[2,:], label="ωy", linewidth=3)
plot!(ω_traj[3,:], label="ωz", linewidth=3)

#savefig("figures/hw1/major_axis_spin.png")


#momentum sphere plots 
#show equilibrium points and several example trajectories

#calculate the momentum 

h_traj = zeros(3, N)

for k = 1:N 

    h_traj[:,k] = J_principle_SI*ω_traj[:,k]

end

h_traj 

#norm of the angular velocity vector 
h_norm = zeros(N)

for k=1:N

    h_norm[k] = norm(h_traj[:,k])

end

h_traj_normalized = zeros(3, N)

for k=1:N

    h_traj_normalized[:,k] = h_traj[:,k]/h_norm[k]

end

#sphere radius
r = 1

# Generate the angles for the mesh grid
θ = range(0, 2π, length=50) 
ϕ = range(0, π, length=50) 

# calculate the Cartesian coordinates
X = [r * sin(phi) * cos(theta) for theta in θ, phi in ϕ]
Y = [r * sin(phi) * sin(theta) for theta in θ, phi in ϕ]
Z = [r * cos(phi) for theta in θ, phi in ϕ]

surface(X, Y, Z, 
        title = "Momentum Sphere", 
        legend = false, 
        colorbar = false,
        camera = (55, 30), xlabel="X", ylabel="Y", zlabel="Z") # Sets the viewing angle


#plot momentum trajectory 
plot!(h_traj_normalized[1,:], h_traj_normalized[2,:], h_traj_normalized[3,:], linewidth=3, linecolor="blue")


#minor axis + 
scatter!([1], [0], [0], markersize=5, markercolor="green")


#intermediate axis +
#scatter!([0], [1], [0], markersize=5, markercolor="red")

#intermediate axis -
#scatter!([0], [-1], [0], markersize=5, markercolor="red")

#major axis - 
#scatter!([0], [0], [-1], markersize=5, markercolor="green")

#minor axis -
#scatter!([-1], [0], [0], markersize=5, markercolor="green")

#major axis +
#scatter!([0], [0], [1], markersize=5, markercolor="green")

#savefig("figures/hw1/minor_axis_momentum_traj.png")


#simulation for HW2 Problem 1.5
#Calculate wheel momentum for dynamic balance and simulation about the solar panel normal

#spin rate 
ωs_magnitude = desired_rad_s 

#desired spin axis 
ωs_direction = solar_normal 

#desired spin vector 
ωs = ωs_magnitude*ωs_direction


#maybe change this 

#inertia about the desired spin axis (this is working)
#Js = (ωs/norm(ωs))'*J_b_SI*(ωs/norm(ωs))
#ρs = ωs_magnitude*(1.2*J_b_SI[3,3] - Js)
#desired spin vector on the left
#ρ0 = [ωs'; hat(ωs)]\[ρs*ωs_magnitude; -hat(ωs)*J_b_SI*ωs]

#equations for the perturbed inertia
Js = (ωs/norm(ωs))'*J_b_SI_perturbed*(ωs/norm(ωs))

#we get this from the superspin condition
ρs = ωs_magnitude*(1.2*J_b_SI_perturbed[3,3] - Js)

#desired spin vector on the left
ρ0 = [ωs'; hat(ωs)]\[ρs*ωs_magnitude; -hat(ωs)*J_b_SI_perturbed*ωs]




#simulate with this rotor speed
ω0_2 = ωs 
u0_2 = zeros(3)
ω_traj_2 = zeros(nx_ω + 3,N)
ω0_2_perturbed = ωs + 0.1*randn(3)
ω_traj_2[:,1] = [ω0_2_perturbed; ρ0]
#simulate about each principle axis
for k=1:N-1 

    ω_traj_2[:,k+1] = RK4_integrator_wcontrol(euler_dynamics_body, ω_traj_2[:,k], u0_2)
    
end

plot(ω_traj_2[1,:], label="ωx", title="Simualtion about ωs", linewidth=3)
plot!(ω_traj_2[2,:], label="ωy", linewidth=3)
plot!(ω_traj_2[3,:], label="ωz", linewidth=3)

#savefig("figures/hw2/solar_panel_normal_spin.png")

#HW 2 Section 2 Simulation

#no rotor momentum rate control
orbit_traj_combined = zeros(16, total_N) 

orbit_traj_combined[:,1] = [eci0_km; [1,0,0,0]; ω0_2_perturbed; ρ0]

for k=1:total_N-1
    #using the wrong dynamics here...
    #orbit_traj_combined[:,k+1] = RK4_integrator_wcontrol(orbit_attitude_dynamics, orbit_traj_combined[:,k], zeros(3))

    orbit_traj_combined[:,k+1] = RK4_integrator_fullsim(orbit_attitude_dynamics, orbit_traj_combined[:,k], zeros(3))

end

q_traj = orbit_traj_combined[7:10,:]

plot(q_traj[1,:], label="q1", title="Attitude Quaternion", linewidth = 3)
plot!(q_traj[2,:], label="q2", linewidth = 3)
plot!(q_traj[3,:], label="q3", linewidth = 3)
plot!(q_traj[4,:], label="q4", linewidth = 3)

#norm(q_traj[:,600])

#savefig("figures/hw2/attitude_quaternion_updated.png")

#solar normal in the body frame
solar_normal

solar_normal_ECI_traj = zeros(3, total_N)

#use the quaternion trajectory to rotate a vector 
#solar normal in the body frame is fixed

for i=1:total_N
    solar_normal_ECI_traj[:,i] = Q(q_traj[:,i])*solar_normal

end


plot(solar_normal_ECI_traj')

angle_diff = zeros(total_N)

for i=1:total_N 

    angle_diff[i] = acosd(dot(solar_normal_ECI_traj[:,i], [1,0,0]))

end

solar_normal_ECI_traj

plot(angle_diff, ylim = [0, 180], label="pointing error", title="Pointing Error", ylabel="Degrees", linewidth=3)

#savefig("figures/hw2/pointing_error_updated.png")
#HW 2 section 3 
function generate_noisy_measurements(std_dev, N_measurements)


    #bearing vector measurements in the body frame 
    b_measurements = zeros(3, N_measurements)

    #bearing vector mea
    noisy_b_measurements = zeros(3,N_measurements)

    #convert the degrees standard deviation into radians
    std_dev_rad = deg2rad(std_dev)

    #get the radians
    var_rad = std_dev_rad^2

    all_P = zeros(3,3,N_measurements)

    for i =1:N_measurements

        direction = randn(3)

        b_measurements[:,i] = direction/norm(direction)

        #last term is for numerical reasons so that we can take the cholesky decomp 
        all_P[:,:,i] = var_rad*(I - b_measurements[:,i]*b_measurements[:,i]') + (1e-15 * Matrix(1.0I, 3,3))

        #sample noise
        #direction is now the mean 
        noisy_b_measurements[:,i] = b_measurements[:,i] + cholesky(all_P[:,:,i]).L*randn(3)

        noisy_b_measurements[:,i] = noisy_b_measurements[:,i]/norm(noisy_b_measurements[:,i])

    end

    return b_measurements, noisy_b_measurements, all_P

end

b_measurements, noisy_b_measurements, all_P = generate_noisy_measurements(0.002, 100)

#check error statistics

angle_diff = zeros(100)

for i=1:100

    angle_rad = acos(dot(b_measurements[:,i], noisy_b_measurements[:,i]))

    angle_diff[i] = rad2deg(angle_rad)

end

scatter([angle_diff], title="Angle Difference", xlabel="Measurement Number", ylabel = " Deviation (Degrees) ", label="angle difference")

#calculate the mean from all these angle differences 
mean = sum(angle_diff)/100

plot!(ones(100)*mean, linewidth=3, label="average")

#savefig("figures/hw2/measurement_error.png")


#HW2 part 4
#using these measurements, perform static attitude determination

#input: n_b_m is a set of noisy body measurements
# i_m are the measurements in the intertial frame

#N is the number of measurements 
function solve_wabha_sdp(n_b_m, i_m, N_measurements)

    #create the attitude profile matrix 
    w = ones(N_measurements)

    B = zeros(3,3)

    for k = 1:N_measurements 

        B += w[k]*n_b_m[:,k]*i_m[:,k]'

    end

    Q = Variable(3,3)
    prob = maximize(tr(B*Q), [Matrix(1.0*I, 3,3) Q'; Q Matrix(I, 3,3)] ⪰ 0)

    solve!(prob, Clarabel.Optimizer)

    return Q.value

end

function solve_wabha_svd(n_b_m, i_m, N_measurements)

    #create the attitude profile matrix 
    w = ones(N_measurements)

    B = zeros(3,3)

    for k = 1:N_measurements 

        B += w[k]*n_b_m[:,k]*i_m[:,k]'

    end

    F = svd(B)

    Q = F.V*F.U'

    return Q 

end

function evaluate_accuracy(Q_true, Q_estimate)

    error_deg = (180/pi)*norm(unhat(log(Q_estimate'*Q_true)))

    return error_deg
end

#come up with a random attitdue 
#Qtrue = exp(hat(randn(3)))

#make inertial measurements
#input: true attitude and body measurments (no noise)

function create_inertial_measurements(Q_true, b_m, N_measurements)

    i_m = zeros(3, N_measurements)

    for i=1:N_measurements

        i_m[:,i] = Q_true*b_m[:,i]

    end

    return i_m

end

function montecarlo_setup(N_measurements)

    b_measurements, noisy_b_measurements, all_P = generate_noisy_measurements(0.002, N_measurements)

    #come up with a random attitdue 
    Q_true = exp(hat(randn(3)))

    i_m = create_inertial_measurements(Q_true, b_measurements, N_measurements)

    Q_sdp = solve_wabha_sdp(noisy_b_measurements, i_m, N_measurements)

    Q_svd = solve_wabha_svd(noisy_b_measurements, i_m, N_measurements)

    error_deg_sdp = evaluate_accuracy(Q_true, Q_sdp)

    error_deg_svd = evaluate_accuracy(Q_true, Q_svd)

    return error_deg_sdp, error_deg_svd

end


montecarlo_trials = 50
sdp_error = zeros(montecarlo_trials)
svd_error = zeros(montecarlo_trials)
t_sdp = zeros(montecarlo_trials) 
t_svd = zeros(montecarlo_trials)


for i=1:montecarlo_trials

    sdp_error[i], svd_error[i] = montecarlo_setup(100)

end

#plot(sdp_error, linewidth=3, title="SDP Wabha Rotation Error", xlabel="MonteCarlo Run", ylabel="Error (degrees)")

#savefig("figures/hw2/sdp_wabha_error.png")

plot(svd_error, linewidth=3, title="SVD Wabha Rotation Error", xlabel="MonteCarlo Run", ylabel="Error (degrees)")

#savefig("figures/hw2/svd_wabha_error.png")


#run a timing simulation 
b_measurements, noisy_b_measurements, all_P = generate_noisy_measurements(0.002, 100)

#come up with a random attitdue 
Q_true = exp(hat(randn(3)))

i_m = create_inertial_measurements(Q_true, b_measurements, 100)

N_measurements = 100

#get timing results 
t_sdp = @belapsed solve_wabha_sdp($noisy_b_measurements, $i_m, $N_measurements)

t_svd = @belapsed solve_wabha_svd($noisy_b_measurements, $i_m, $N_measurements)

sdp_error 
svd_error 

plot(sdp_error )

plot!(svd_error)

#montecarlo_setup(1000)

    
    