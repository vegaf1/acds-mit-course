using Pkg
Pkg.activate(".")

using LinearAlgebra 
using SatelliteDynamics 
using Plots 

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

principle_axes = J_b_eig.vectors 

J_b_eig.values 
#moment of inertia about principle axes
J_principle = principle_axes'*J_b*principle_axes 

#get into units of meters and seconds
J_principle_SI = J_principle/(100^2)
 
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

