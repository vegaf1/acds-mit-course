using Pkg
Pkg.activate(".")

using LinearAlgebra 
import SatelliteDynamics as SD
using Plots 
using Convex
using Clarabel
using BenchmarkTools

include("../src/dynamics.jl") 
include("../src/helper.jl")
include("../src/integrators.jl")
include("../src/spacecraft_model.jl")
include("../src/measurement_models.jl")
include("../src/static_attitude_determination.jl")


#simulation for HW2 Problem 1.5
#Calculate wheel momentum for dynamic balance and simulation about the solar panel normal
#this sim is in the body frame 
#state size for omega
nx_ω= 3

#omega sim timestep (10 Hz)
dt_ω = 0.1

#number of timesteps for ω sim (around 3 minutes at that rate)
N_ω = 2000

#desired rpm (rotations per minute)
desired_rpm = 10 

#convert rpm to rad/s 
desired_rad_s = desired_rpm*(2*pi/60)

#spin rate 
ωs_magnitude = desired_rad_s 

#desired spin axis 
ωs_direction = solar_normal 

#desired spin vector 
ωs = ωs_magnitude*ωs_direction

#equation for the perturbed inertia
Js = (ωs/norm(ωs))'*J_b_SI_perturbed*(ωs/norm(ωs))

#we get this from the superspin condition
ρs = ωs_magnitude*(1.2*J_b_SI_perturbed[3,3] - Js)

#rotor momentum for dynamic balance (spinning about solar normal)
ρ0 = [ωs'; hat(ωs)]\[ρs*ωs_magnitude; -hat(ωs)*J_b_SI_perturbed*ωs]

#simulate with this rotor speed in the body frame
ω0_2 = ωs 
u0_2 = zeros(3)
ω_traj_2 = zeros(nx_ω + 3,N_ω)
ω0_2_perturbed = ωs + 0.1*randn(3)
ω_traj_2[:,1] = [ω0_2_perturbed; ρ0]

#simulate about each solar panel normal vector using dynamic balance
for k=1:N_ω-1 

    ω_traj_2[:,k+1] = RK4_integrator_wcontrol(euler_dynamics_body, ω_traj_2[:,k], u0_2, dt_ω)
    
end

all_time = range(0, step=dt_ω, length=2000)


plot(all_time, ω_traj_2[1,:], label="ωx", title="Simulation about ωs", linewidth=3, xlabel = "Time (s)", ylabel = "Angular Velocity (rad/s)")
plot!(all_time, ω_traj_2[2,:], label="ωy", linewidth=3)
plot!(all_time, ω_traj_2[3,:], label="ωz", linewidth=3)

#savefig("figures/hw2/solar_panel_normal_spin_updated.png")

#spacecraft dynamics sim
#no rotor momentum rate control

#Define the orbit initial condition and discretization
#initial orbit (osculating orbital elements)
oe0  = [SD.R_EARTH + 550e3, 0.0, 90.0, 0, 0, 0]

#period (based on the semimajor axis)
T = SD.orbit_period(oe0[1])

#convert to cartesian (in meters)
eci0 = SD.sOSCtoCART(oe0, use_degrees=true)

#scale to km and km/s 
eci0_km = eci0./1000

#define the orbit trajectory 
#state = [position; velocity attitude; angular velocity; rotor momentum]
orbit_traj_combined = zeros(16, N_ω) 

orbit_traj_combined[:,1] = [eci0_km; [1,0,0,0]; ω0_2_perturbed; ρ0]

#simulating both the position and attitude at 10 Hz. (3 minutes worth of data)
for k=1:N_ω-1

    #assuming no momentum rate control
    orbit_traj_combined[:,k+1] = RK4_integrator_fullsim(orbit_attitude_dynamics, orbit_traj_combined[:,k], zeros(3), dt_ω)

end

q_traj = orbit_traj_combined[7:10,:]

plot(all_time, q_traj[1,:], label="q1", title="Attitude Quaternion", linewidth = 3, xlabel="Time (s)", ylabel="Value")
plot!(all_time, q_traj[2,:], label="q2", linewidth = 3)
plot!(all_time, q_traj[3,:], label="q3", linewidth = 3)
plot!(all_time, q_traj[4,:], label="q4", linewidth = 3)

#savefig("figures/hw2/attitude_quaternion_v3.png")

solar_normal_ECI_traj = zeros(3, N_ω)

#use the quaternion trajectory to rotate a vector 
#solar normal in the body frame is fixed
for i=1:N_ω
    solar_normal_ECI_traj[:,i] = Q(q_traj[:,i])*solar_normal

end

plot(solar_normal_ECI_traj')

#calculate the angle difference between the eci solar panel normal vector and 
#the x axis (assuming sun is at the x axis)
angle_diff = zeros(N_ω)
for i=1:N_ω 

    angle_diff[i] = acosd(dot(solar_normal_ECI_traj[:,i], [1,0,0]))

end

plot(all_time, angle_diff, ylim = [0, 180], label="pointing error", title="Pointing Error", ylabel="Degrees", linewidth=3, xlabel="Time (s)")

#savefig("figures/hw2/pointing_error_v3.png")

#HW 2 section 3 
#generate noisy bearing measurements along with the ground truth 
#and covariance matrices


#from the blue canyon 6u cubesat spec sheet 
pointing_accuracy = 0.002
N_measurements = 100

b_measurements, noisy_b_measurements, all_P = generate_noisy_bearing_measurements(pointing_accuracy, N_measurements)

#check error statistics
angle_diff = zeros(N_measurements)

for i=1:N_measurements

    angle_rad = acos(dot(b_measurements[:,i], noisy_b_measurements[:,i]))

    angle_diff[i] = rad2deg(angle_rad)

end

scatter([angle_diff], title="Angle Difference", xlabel="Measurement Number", ylabel = " Deviation (Degrees) ", label="angle difference")

#calculate the mean from all these angle differences 
mean = sum(angle_diff)/N_measurements

plot!(ones(N_measurements)*mean, linewidth=3, label="average")

#savefig("figures/hw2/measurement_error.png")

#HW2 part 4 
function montecarlo_setup(N_measurements)

    b_measurements, noisy_b_measurements, all_P = generate_noisy_bearing_measurements(pointing_accuracy, N_measurements)

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
b_measurements, noisy_b_measurements, all_P = generate_noisy_bearing_measurements(pointing_accuracy, N_measurements)

#come up with a random attitdue 
Q_true = exp(hat(randn(3)))

i_m = create_inertial_measurements(Q_true, b_measurements, N_measurements)

#get timing results 
t_sdp = @belapsed solve_wabha_sdp($noisy_b_measurements, $i_m, $N_measurements)

t_svd = @belapsed solve_wabha_svd($noisy_b_measurements, $i_m, $N_measurements)

sdp_error 
svd_error 

plot(sdp_error)

plot!(svd_error)