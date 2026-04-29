using Pkg
Pkg.activate(".")

using LinearAlgebra 
import SatelliteDynamics as SD  
using Plots 
using Convex
using Clarabel
using BenchmarkTools
using SatelliteToolboxGeomagneticField
using SatelliteToolboxTransformations
using BlockDiagonals

include("../src/dynamics.jl") 
include("../src/helper.jl")
include("../src/integrators.jl")
include("../src/spacecraft_model.jl")
include("../src/measurement_models.jl")
include("../src/control.jl")
include("../src/mekf.jl")

#Part 1
#calculate the thruster and wheel jacobians jacobian 
Bt, Bw = actuator_jacobians()

#Part 2 
#Calculate Disturbances

#define an initial epoch 
epc0 = SD.Epoch(2012, 11, 8, 12, 0, 0, 0.0)

#orbital dynamics simulation 
#initial orbit (osculating orbital elements)
oe0  = [SD.R_EARTH + 550e3, 0.0, 90.0, 0, 0, 0]

#period (based on the semimajor axis)
period = SD.orbit_period(oe0[1])

#convert to cartesian (in meters)
eci0 = SD.sOSCtoCART(oe0, use_degrees=true)

#scale to km and km/s 
eci0_km = eci0./1000

#number of discrete points per orbit
N = 50000

#integration timestep (~10 Hz)
dt_orbit = period/(N-1)

dt = dt_orbit

#state size 
nx = 16

#number of revs to simulate 
n_revs = 1

#N-1 to not repeat the same starting point every rev 
total_N = (N-1)*n_revs

#create a state trajectory to save the result
x_traj = zeros(nx, total_N)

#standard deviation of 2 degrees per second 
ω0 = randn(3)*(3.0 * pi/180)
ρ0 = zeros(3)

#initial attitude
q0 = [1,0,0,0]

#set the initial condtition 
x_traj[:,1] = [eci0_km; q0; ω0; ρ0]

#time series
horizon = LinRange(0, period*n_revs, total_N)

#simulate forward with no controls
for k=1:total_N-1

    x_traj[:,k+1] = RK4_integrator_fullsim_wtime(orbit_attitude_dynamics_v2, x_traj[:,k], zeros(7), dt_orbit, horizon[k])

end

horizon 
position_trajectory = x_traj[1:3, :]
velocity_trajectory = x_traj[4:6, :]
attitude_trajectory = x_traj[7:10, :]
angular_velocity_trajectory = x_traj[11:13, :]



plot(x_traj[1,:], x_traj[2,:], x_traj[3,:], title = "Orbit Trajectory", label = "position")
xlabel!("X [km]") 
ylabel!("           Y [km]")
zlabel!("Z [km ]")

horizon 
#find the maximum of each torque

total_N 
τ_gg = zeros(3, total_N)
τ_drag = zeros(3, total_N)

all_time_days = horizon/86400

horizon 
for i=1:total_N
    epc = epc0 + horizon[i]
    #position argument needs to be in meters
    ρ_atm = SD.density_harris_priester(epc, position_trajectory[:,i]*1000)
    τ_gg[:,i] = gravity_gradient_torque(x_traj[:,i])
    τ_drag[:,i] = drag_torque(x_traj[:,i], ρ_atm, 2.0)

end

plot(all_time_days, τ_gg[1,:], label= "τx", linewidth = 3, title = "Gravity Gradient Torque", xlabel="Days", ylabel="Torque (Nm)")
plot!(all_time_days, τ_gg[2,:], label= "τy", linewidth = 3)
plot!(all_time_days, τ_gg[3,:], label= "τz", linewidth = 3)


plot(all_time_days, τ_drag[1,:], label= "τx", linewidth = 3, title = "Drag Torque", xlabel="Days", ylabel="Torque (Nm)")
plot!(all_time_days, τ_drag[2,:], label= "τy", linewidth = 3)
plot!(all_time_days, τ_drag[3,:], label= "τz", linewidth = 3)

#find the maximum magnitude

τ_drag_norm = zeros(total_N)
τ_gg_norm = zeros(total_N)

for i=1:total_N

    τ_drag_norm[i] = norm(τ_drag[:,i])
    τ_gg_norm[i] = norm(τ_gg[:,i])

end

maximum(τ_drag_norm) 

maximum(τ_gg_norm) 

#simulate without the environmental torques and check the difference in attitude and angular velocity 
orbit_traj_combined = zeros(nx, total_N)

#set the initial condtition 
orbit_traj_combined[:,1] = [eci0_km; [1,0,0,0]; ω0; ρ0]

for k=1:total_N-1

    #assuming no momentum rate control
    orbit_traj_combined[:,k+1] = RK4_integrator_fullsim(orbit_attitude_dynamics, orbit_traj_combined[:,k], zeros(3), dt_orbit)

end

#state trajectories with no perturbations. no drag acceleration/torques and gravity gradient torques 
position_trajectory_np = orbit_traj_combined[1:3, :]
velocity_trajectory_np = orbit_traj_combined[4:6, :]
attitude_trajectory_np = orbit_traj_combined[7:10, :]
angular_velocity_trajectory_np = orbit_traj_combined[11:13, :]


position_difference = position_trajectory - position_trajectory_np 

norm(position_difference[:,end])

plot(position_trajectory_np[1,:], position_trajectory_np[2,:], position_trajectory_np[3,:])

#difference_angular_velocity = angular_velocity_trajectory - angular_velocity_trajectory_np
difference_angular_velocity_norm = zeros(total_N)
difference_attitude = zeros(total_N)

for i =1:total_N
    difference_angular_velocity_norm[i] = norm(angular_velocity_trajectory[:,i] - angular_velocity_trajectory_np[:,i])

    #get the vector part of the quaternion as the delta rotation (in radians)
    difference_attitude[i] = norm(H'*(L(attitude_trajectory[:,i])'*(attitude_trajectory_np[:,i])))

end

plot(all_time_days, difference_angular_velocity_norm, title="Angular Velocity Difference", xlabel="Days", ylabel="ω Difference (rad/s)", label="Δω")
plot(all_time_days, difference_attitude, title="Attitude Difference", xlabel="Days", ylabel="Attitude Difference (rad)", label=false)

#calculate the average of the disturbance torques over 1 orbit

total_drag = zeros(3)
total_gg = zeros(3)

for i=1:N-1
    total_drag += τ_drag[:,i]
    total_gg += τ_gg[:,i]
end

avg_drag_one_orbit = total_drag/(N-1)
avg_gg_one_orbit = total_gg/(N-1)

#maximum change in angular momentum 
#h_dot = torque -> delta_h = ∑torque*dt
#approximately 15 orbits in 1 day
n_orbits_per_day =15

ΔH_max_day = sum(norm.(eachcol(τ_gg[:, 1:N] + τ_drag[:, 1:N]))) * dt_orbit * n_orbits_per_day

ΔH_max_day_orbit = ΔH_max_day/n_orbits_per_day

rw_limit = 0.015/ΔH_max_day_orbit

#Part 3 Attitude Regulation
# Qn = Matrix(1.0*I, 9,9).*[100*ones(3); 10*ones(3); 0.01*ones(3)]

#allow the wheels to hold momentum by penalizing them small
#this combo is working for high torques
# Q_lqr = Matrix(1.0*I, 9,9).*[100*ones(3); 10*ones(3); 0.01*ones(3)]
# R_lqr = Matrix(1.0*I, 7,7).*[0.01*ones(3); 0.1*ones(4)]

#need to tune for so the wheels torque and thruster torque is not that high...
Qn = Matrix(1.0*I, 9,9).*[10.0*ones(3); 1.0*ones(3); 1.0*ones(3)]
Q_lqr = Matrix(1.0*I, 9,9).*[10*ones(3); 1.0*ones(3); 1.0*ones(3)]
R_lqr = Matrix(1.0*I, 7,7).*[3000*ones(3); 1000*ones(4)]


#cost to go and feedback gains
P_lqr, K_lqr = ricatti_recursion(Qn, Q_lqr, R_lqr, total_N, dt_orbit)

 
x_traj_controlled = zeros(16, total_N)

#initial ground truth 
initial_state = [eci0_km; q0; ω0; ρ0]
initial_bias = generate_bias(dt_orbit, 1) 

initial_bias 
 
x_traj_controlled[:,1] = initial_state
#generate a desired attitude +-90 degrees from the ground truth 
#sample a random rotation from identity to track 
#axis can point in any direction 
axis = randn(3)
axis /= norm(axis)

#check this
θ = rand()*(90 * pi/180)
#this is the desired quaternion used (in the range of +- 90 from the ground truth)
#check division by 2 
# qd = L(q0)*expq(θ/2*axis)
qd = L(q0)*expq(θ*axis)

#mekf is initially around 10 degrees off from the ground truth
#LQR will be on the MEKF state 
ϕ0 = (20*pi/180)*randn(3) 
#sample a random small rotation and apply it to the ground truth initial state (make it consistent with V)
q0_mekf = L(q0)*expq(ϕ0)
#initial bias estimate from the ground truth (make it consistent with V) 1e-5 works
b0_mekf = initial_bias[:,1]+(1e-5*randn(3))


#initialize covariance from hw3
P0 = Matrix(1.0*I, 6,6).*[abs.(ϕ0); ones(3)*1e-5]

#P0 = Matrix(1.0*I, 6,6).*[ϕ0.^2; ones(3)*(1e-5).^2]



mekf_state = zeros(7, total_N)
mekf_P = zeros(6,6, total_N)
kalman_gain = zeros(6,12, total_N)

#initialize mekf state 
mekf_state[:,1] = [q0_mekf; b0_mekf]
mekf_P[:,:,1] = P0


#initialize 
#noisy_gyro_measurement = zeros(3)
controls = zeros(7, total_N-1)
error_states = zeros(9, total_N-1)

#define constants and noise
_, _, _, sun_sensor_specs, mag_sensor_specs, M_gyro, W_st, _ = get_measurement(x_traj_controlled[:,1], [epc0+horizon[1]], initial_bias[:,1])
W_mekf = BlockDiagonal([Matrix(1.0*I,3,3)*7.61e-5, Matrix(1.0*I,3,3)*0.0016, W_st, W_st])
V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*1e-6, Matrix(1.0*I, 3,3)*2.58e-8])

noisy_gyro_measurementz = zeros(3)

#running true gyro bias state for the random walk
true_bias_state = initial_bias[:,1]

all_bias_z = zeros(3, total_N) 

all_bias_z[:,1] = initial_bias

for k=1:total_N-1

    #generate a noisy gyro measurement (only for timestep 1)
    if k==1
        gyro_mat, _ = generate_gyro_measurements(reshape(ω0,3,1), initial_bias, dt_orbit)
        noisy_gyro_measurementz .= gyro_mat[:,1]
    end

    #print(noisy_gyro_measurementz)
    #generate a control input using the error state
    #x = [ϕ; ω; ρ]

    #pass in mekf quaternion estimate, desired quaternion, gyro measurement, true ρ
    error_states[:,k] = error_state(mekf_state[1:4, k], qd, noisy_gyro_measurementz- mekf_state[5:7, k], x_traj_controlled[14:16,k])

    #control using the mekf state
    controls[:,k] = -K_lqr[:,:,k]*error_states[:,k]

    #controls[:,k] = zeros(7)

    #save gyro at time k 
    gyro_at_k = noisy_gyro_measurementz

    #simulate this control
    #pass in time in seconds from start
    x_traj_controlled[:,k+1] = RK4_integrator_fullsim_wtime(orbit_attitude_dynamics_v2, x_traj_controlled[:,k], controls[:,k], dt_orbit, horizon[k])


    #hw 3 sim. trying to find the bug 
    #x_traj_controlled[:,k+1] = RK4_integrator_fullsim(orbit_attitude_dynamics, x_traj_controlled[:,k], zeros(3), dt_orbit)


    #need to get measurements at updated timestep k+1 from ground truth
    #pass in epoch
    #this is gyro at k+1

    noisy_body_measurements, inertial_measurements, noisy_gyro_measurementz, _, _, _, _, all_bias_z[:,k+1] = get_measurement(x_traj_controlled[:,k+1], [epc0+horizon[k+1]], all_bias_z[:,k])

    #mekf step to k+1: propagate using gyro at time k, update with measurements at k+1
    mekf_state[:,k+1], mekf_P[:,:,k+1], _, kalman_gain[:,:,k+1] = mekf_step(mekf_state[:,k], gyro_at_k, mekf_P[:,:,k],
                                                        noisy_body_measurements, inertial_measurements,
                                                        dt_orbit, V_mekf, W_mekf, sun_sensor_specs, mag_sensor_specs, M_gyro)

end

#check the measurements...
# plot(all_bias_z[3,1:2000])
# all_noisy_body_measurements = zeros(14, total_N)
# all_inertial_measurements = zeros(6, total_N)
# all_gyro_measurements = zeros(3, total_N)
# all_bias = zeros(3,total_N)
# all_bias[:,1] = initial_bias[:,1]
# x_traj_controlled 
# for k=1:total_N-1
#     all_noisy_body_measurements[:,k+1], all_inertial_measurements[:,k+1], all_gyro_measurements[:,k+1], _, _, _, _, all_bias[:,k+1] = get_measurement(x_traj_controlled[:,k+1], [epc0+horizon[k+1]], all_bias[:,k])
# end
# #plot(all_gyro_measurements[1,1:2000])
# plot(all_noisy_body_measurements[10,1:2000])
# plot(all_bias[3,:])

q_controlled = x_traj_controlled[7:10,:]
ω_controlled = x_traj_controlled[11:13,:]
ρ_controlled = x_traj_controlled[14:16,:]

#true attitude error over time (degrees)
attitude_error_deg = zeros(total_N)
for i = 1:total_N
    #error quaternion
    q_err = L(qd)' * q_controlled[:, i]

    #norm of axis angle is the rotation in degrees
    attitude_errror_deg[i] = norm(2*H'*q_err)*(180/pi)

    #attitude_error_deg[i] = 2 * norm(H' * q_err) * (180/pi)
end

#wheel torque limit (Nm) (L2 norm)
wheel_ρ_dot_limit = 0.042

#thruster force limit (per axis)
thruster_force_limit = 0.015


attitude_error_deg

qd 

plot(attitude_error_deg, title="Attitude Error", xlabel="Days", ylabel="Error (deg)", label=false)

attitude_error_deg 
plot(q_controlled[1,1:2000])
plot!(mekf_state[1,1:2000])

plot(q_controlled[2,1:2000])
plot!(mekf_state[2,1:2000])

plot(q_controlled[3,1:2000])
plot!(mekf_state[3,1:2000])

plot(q_controlled[4,1:2000])
plot!(mekf_state[4,1:2000])

controls[1:3,:]

plot(controls[1,:])
plot(controls[2,:])
plot(controls[3,:])


#plots for the control
plot(all_time_days[1:total_N-1], controls[1,:], linewidth = 3, label="wheel torque (x)", title="Wheel Torque (X)")
plot!(all_time_days[1:total_N-1], wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque upper limit")
plot!(all_time_days[1:total_N-1], -wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque lower limit")

plot(all_time_days[1:total_N-1], controls[2,:],linewidth = 3, label="wheel torque (y)", title="Wheel Torque (Y)")
plot!(all_time_days[1:total_N-1], wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque upper limit")
plot!(all_time_days[1:total_N-1], -wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque lower limit")

plot(all_time_days[1:total_N-1], controls[3,:], linewidth = 3, label="wheel torque (z)", title="Wheel Torque (Z)")
plot!(all_time_days[1:total_N-1], wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque upper limit")
plot!(all_time_days[1:total_N-1], -wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque lower limit")

plot(all_time_days[1:total_N-1], controls[4,:], linewidth=3, label="thruster 1 force", title="Thruster 1 Force")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")

plot(all_time_days[1:total_N-1], controls[5,:], linewidth=3, label="thruster 2 force", title="Thruster 2 Force")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")


plot(all_time_days[1:total_N-1], controls[6,:], linewidth=3, label="thruster 3 force", title="Thruster 3 Force")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")


plot(all_time_days[1:total_N-1], controls[7,:], linewidth=3, label="thruster 4 force", title="Thruster 4 Force")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")


#Angular Velocity Error State
plot(all_time_days[1:total_N-1], ω_controlled[1,:], linewidth=3, label=false, title="Angular Velocity", label = "ωx") 
plot!(all_time_days[1:total_N-1], ω_controlled[2,:], linewidth=3, label=false, label= "ωy")
plot!(all_time_days[1:total_N-1], ω_controlled[3,:], linewidth=3, label=false, label= "ωz")

#wheel momentum Error state
plot(all_time_days[1:total_N-1], ρ_controlled[7,:], linewidth=3, label=false, title="Axis Angle Error State", label = "ρx")
plot!(all_time_days[1:total_N-1], ρ_controlled[8,:], linewidth=3, label=false, label= "ρy")
plot!(all_time_days[1:total_N-1], ρ_controlled[9,:], linewidth=3, label=false, label= "ρz")