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
R_lqr = Matrix(1.0*I, 7,7).*[5000*ones(3); 2000*ones(4)]


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
#θ = 0.7*(90*pi/180)
#this is the desired quaternion used (in the range of +- 90 from the ground truth)
qd = L(q0)*expq(θ/2*axis)
#qd = L(q0)*expq(θ*axis)

#mekf is initially around 10 degrees off from the ground truth
#LQR will be on the MEKF state 
ϕ0 = (20*pi/180)*randn(3) 
#sample a random small rotation and apply it to the ground truth initial state (make it consistent with V)
q0_mekf = L(q0)*expq(ϕ0/2)
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
    attitude_error_deg[i] = norm(2*H'*q_err)*(180/pi)

    #attitude_error_deg[i] = 2 * norm(H' * q_err) * (180/pi)
end

#wheel torque limit (Nm) (L2 norm)
wheel_ρ_dot_limit = 0.042

#thruster force limit (per axis)
thruster_force_limit = 0.015


attitude_error_deg[100]

plot(attitude_error_deg[1:200])

qd 

plot(all_time_days[1:total_N],attitude_error_deg, title="Attitude Error", xlabel="Days", linewidth=3, ylabel="Error (deg)", label=false)

attitude_error_deg

#savefig("figures/hw4/attitude_error_test2.png")

converged_trajectory = attitude_error_deg[500:end] 

mean_converged_trajectory = sum(converged_trajectory)/size(converged_trajectory)[1]

rmse_num = 0

for i = 1:size(converged_trajectory)[1]

    rmse_num += (norm(converged_trajectory[i] - mean_converged_trajectory))^2

end

#rmse calculation for regulator
rmse = sqrt(rmse_num/size(converged_trajectory)[1])


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
plot(all_time_days[1:total_N-1], controls[1,:], linewidth = 3, label="wheel torque (x)", title="Wheel Torque (X)", ylabel="Wheel Torque (Nm)", xlabel="Days")
plot!(all_time_days[1:total_N-1], wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque upper limit")
plot!(all_time_days[1:total_N-1], -wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque lower limit")
savefig("figures/hw4/wheel_torque_x_test2.png")

plot(all_time_days[1:total_N-1], controls[2,:],linewidth = 3, label="wheel torque (y)", title="Wheel Torque (Y)", ylabel="Wheel Torque (Nm)", xlabel="Days")
plot!(all_time_days[1:total_N-1], wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque upper limit")
plot!(all_time_days[1:total_N-1], -wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque lower limit")
savefig("figures/hw4/wheel_torque_y_test2.png")

plot(all_time_days[1:total_N-1], controls[3,:], linewidth = 3, label="wheel torque (z)", title="Wheel Torque (Z)", ylabel="Wheel Torque (Nm)", xlabel="Days")
plot!(all_time_days[1:total_N-1], wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque upper limit")
plot!(all_time_days[1:total_N-1], -wheel_ρ_dot_limit*ones(total_N-1), linewidth = 3, label="wheel torque lower limit")
savefig("figures/hw4/wheel_torque_z_test2.png")


plot(all_time_days[1:total_N-1], controls[4,:], linewidth=3, label="thruster 1 force", title="Thruster 1 Force", ylabel="Thruster Force (N)", xlabel="Days")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")
savefig("figures/hw4/force_thruster_1_test2.png")

plot(all_time_days[1:total_N-1], controls[5,:], linewidth=3, label="thruster 2 force", title="Thruster 2 Force", ylabel="Thruster Force (N)", xlabel="Days")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")
savefig("figures/hw4/force_thruster_2_test2.png")

plot(all_time_days[1:total_N-1], controls[6,:], linewidth=3, label="thruster 3 force", title="Thruster 3 Force", ylabel="Thruster Force (N)", xlabel="Days")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")
savefig("figures/hw4/force_thruster_3_test2.png")

plot(all_time_days[1:total_N-1], controls[7,:], linewidth=3, label="thruster 4 force", title="Thruster 4 Force", ylabel="Thruster Force (N)", xlabel="Days")
plot!(all_time_days[1:total_N-1], thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster upper limit")
plot!(all_time_days[1:total_N-1], -thruster_force_limit*ones(total_N-1), linewidth = 3, label="thruster lower limit")
savefig("figures/hw4/force_thruster_4_test2.png")

#Angular Velocity Error State
plot(all_time_days[1:total_N], ω_controlled[1,:], linewidth=3, title="Angular Velocity", label = "ωx", xlabel="Days", ylabel="ω (rad/s)") 
plot!(all_time_days[1:total_N], ω_controlled[2,:], linewidth=3, label= "ωy")
plot!(all_time_days[1:total_N], ω_controlled[3,:], linewidth=3, label= "ωz")
savefig("figures/hw4/omega_test2.png")

#wheel momentum Error state
plot(all_time_days[1:total_N], ρ_controlled[1,:], linewidth=3, title="Wheel Momenta", label = "ρx", xlabel="Days", ylabel="Wheel Momenta")
plot!(all_time_days[1:total_N], ρ_controlled[2,:], linewidth=3, label= "ρy")
plot!(all_time_days[1:total_N], ρ_controlled[3,:], linewidth=3, label= "ρz")
savefig("figures/hw4/wheel_momenta_test2.png")

#Part 4 Eigen Axis Slew 

#perturbing the intial attitude
# θ_ea = rand()*(180 * pi/180)
# axis = randn(3)
# axis /= norm(axis)

# #this is the desired quaternion used (in the range of +- 180 from the ground truth)
# qd_ea = L(q0)*expq(θ*axis)

# #get the error quaternion. should be the same axis as above..
# ϕ_error =  2*H'*L(qd_ea)'*q0

# r_axis = ϕ_error/norm(ϕ_error)

# #have to choose either only wheels or thrusters
# function inverse_dynamics(\)

#     #if we only use wheels 
#     ρ_dot = -J*ω_dot - cross(ω, J*ω + ρ)

#     #if we only use thrusters
#     τ = J*ω_dot + cross(ω, J*ω)

#     return ρ_dot
#     #return τ

# end

#once we have the open loop control trajectory, we wrap it in a TVLQR controller



#####
#Part 4 Eigen Axis Slew
#versine trajectory helper functions
"""
versine trajectory helper functions. from zero to some desired angle
using to calculate an attitude trajectory for a slew maneuver 
function settles at θfinals at time π diveded by α
t - time 
α - time constant (sets how long the maneuver takes)
θf - final angle
"""
versine_θ(t, θf, α)      = θf/2 * (1 - cos(π/α * t))
versine_θ_dot(t, θf, α)  = θf/2 * (π/α) * sin(π/α * t)
versine_θ_ddot(t, θf, α) = θf/2 * (π/α)^2 * cos(π/α * t)

"""
Compute nominal trajectory for an eigen-axis slew from q_start to q_final
over T_man seconds at timestep dt_step.
Returns q_nom, ω_nom, ρ_nom, u_nom (7D wheel+thruster), t_nom, θ_total, r_hat.
"""
function eigenaxis_slew_nominal(q_start, q_final, T_man, dt_step, J)
    # Relative rotation from q_start to q_final (in start body frame)
    #q_rel = L(q_start)' * q_final

    #errpr quaternion
    #q_e = L(q_final)'*q_start

    #try backward for buidling the forward trajectory
    q_e = L(q_start)'*q_final

    #if q_rel[1] < 0; q_rel = -q_rel; end

    # Eigen-axis and total angle via quaternion log
    #get the error quaternion

    #error state. use log instead of approximation
    #equation from notes may have a bug. loqq already returns a 3d vector
    #ϕ = 2*H'*logq(q_e)

    ϕ = 2*logq(q_e)

    #from this error state, we get the angle and the rotation axis
    θf = norm(ϕ)

    rf = ϕ/norm(ϕ)

    #ϕ_log = norm(q_rel[2:4]) > 1e-10 ? logq(q_rel) : zeros(3)
    #θ_tot = 2 * norm(ϕ_log)
    #r_hat = norm(ϕ_log) > 1e-10 ? ϕ_log / norm(ϕ_log) : [0.0, 0.0, 1.0]

    N_n = Int(round(T_man / dt_step)) + 1
    t_n = collect(range(0.0, T_man, length=N_n))
    #quaternion
    q_n = zeros(4, N_n)
    #angular velocity
    ω_n = zeros(3, N_n)
    #wheel momentum 
    ρ_n = zeros(3, N_n)

    #controls. only using wheels so zeroing out the thrusters
    u_n = zeros(7, N_n - 1) 

    # Build attitude and angular velocity trajectory using versine
    for k = 1:N_n
        θ_t = versine_θ(t_n[k], θf, T_man)
        # expq((θ/2)*r) gives quaternion for rotation of angle θ around r
        q_n[:,k] = L(q_start) * expq((θ_t/2) * rf)
        ω_n[:,k] = versine_θ_dot(t_n[k], θf, T_man) * rf
    end

    # Inverse dynamics: ρ_dot = -J*ω_dot - ω×(J*ω + ρ), integrate to get ρ_nom
    for k = 1:N_n - 1
        ω_dot_k = versine_θ_ddot(t_n[k], θf, T_man) * rf
        ρ_dot_k = -J * ω_dot_k - hat(ω_n[:,k]) * (J * ω_n[:,k] + ρ_n[:,k])
        u_n[1:3, k] = Bw*ρ_dot_k 

        #euler integrator
        ρ_n[:,k+1]  = ρ_n[:,k] + dt_step * ρ_dot_k
    end

    return q_n, ω_n, ρ_n, u_n, t_n, θf, rf
end



# ---- Slew setup: 180° about z-axis ----

q_sl_start = [1.0, 0.0, 0.0, 0.0]

#z axis
#sl_axis    = normalize([0.0, 0.0, 1.0])

#random axis
sl_axis = randn(3)
sl_axis = sl_axis/norm(sl_axis)

#pi/2 along z axis
q_sl_final = expq((π/2) * sl_axis)    # expq((π/2)*r) = [0; r] = 180° rotation

#in seconds
T_sl = 300.0

q_nom_sl, ω_nom_sl, ρ_nom_sl, u_nom_sl, t_nom_sl, θ_sl_tot, r_sl_hat =
    eigenaxis_slew_nominal(q_sl_start, q_sl_final, T_sl, dt, J_b_SI_perturbed)

N_sl = length(t_nom_sl)

#design a tvlqr controller linearized along the trajectory 
P,K = linear_attitude_dynamics_tvlqr(Qn, Q_lqr, R_lqr, N_sl, dt, ω_nom_sl, J_b_SI_perturbed, ρ_nom_sl)

K_sl = K_lqr[:,:,1]   # steady-state LQR gain (converged from Part 3 Riccati recursion)

x_sl    = zeros(16, N_sl)
mk_sl   = zeros(7,  N_sl)
Pk_sl   = zeros(6, 6, N_sl)
u_sl    = zeros(7,  N_sl - 1)
bias_sl = zeros(3,  N_sl)

ϕ0_sl      = (5 * π/180) * randn(3)
b0_sl      = generate_bias(dt, 1)[:,1]
x_sl[:,1]    = [eci0_km; q_sl_start; zeros(3); zeros(3)]
mk_sl[:,1]   = [L(q_sl_start) * expq(ϕ0_sl); b0_sl]
Pk_sl[:,:,1] = Matrix(1.0*I, 6,6) .* [abs.(ϕ0_sl); ones(3)*1e-5]
bias_sl[:,1] = b0_sl

_, _, _, ss_sl, ms_sl, Mg_sl, Wst_sl, _ = get_measurement(x_sl[:,1], [epc0], b0_sl)
W_m_sl = BlockDiagonal([Matrix(1.0*I,3,3)*7.61e-5, Matrix(1.0*I,3,3)*0.0016, Wst_sl, Wst_sl])
V_m_sl = BlockDiagonal([Matrix(1.0*I,3,3)*1e-6,    Matrix(1.0*I,3,3)*2.58e-8])

gyro_sl = zeros(3)

for k = 1:N_sl - 1
    if k == 1
        gm, _ = generate_gyro_measurements(reshape(zeros(3),3,1), reshape(b0_sl,3,1), dt)
        gyro_sl .= gm[:,1]
    end

    q_mk  = mk_sl[1:4, k];  b_mk  = mk_sl[5:7, k]
    ω_est = gyro_sl - b_mk;  ρ_k   = x_sl[14:16, k]

    # Error state relative to nominal trajectory
    δx = error_state(q_mk, q_nom_sl[:,k], ω_est - ω_nom_sl[:,k], ρ_k - ρ_nom_sl[:,k])

    # Total control = inverse-dynamics feedforward + LQR feedback
    u_sl[:,k] = u_nom_sl[:,k] - K[:,:,k] * δx

    gk = copy(gyro_sl)
    x_sl[:,k+1] = RK4_integrator_fullsim_wtime(
        orbit_attitude_dynamics_v2, x_sl[:,k], u_sl[:,k], dt, t_nom_sl[k])

    nbm, nim, gyro_sl, _, _, _, _, bias_sl[:,k+1] =
        get_measurement(x_sl[:,k+1], [epc0 + t_nom_sl[k+1]], bias_sl[:,k])

    mk_sl[:,k+1], Pk_sl[:,:,k+1], _, _ = mekf_step(
        mk_sl[:,k], gk, Pk_sl[:,:,k], nbm, nim,
        dt, V_m_sl, W_m_sl, ss_sl, ms_sl, Mg_sl)
end

# ---- Regulator comparison: same IC, no trajectory (Part 4.2) ----

x_rg    = zeros(16, N_sl)
mk_rg   = zeros(7,  N_sl)
Pk_rg   = zeros(6, 6, N_sl)
u_rg    = zeros(7,  N_sl - 1)
bias_rg = zeros(3,  N_sl)

x_rg[:,1]    = x_sl[:,1]
mk_rg[:,1]   = mk_sl[:,1]
Pk_rg[:,:,1] = Pk_sl[:,:,1]
bias_rg[:,1] = b0_sl

gyro_rg = zeros(3)

for k = 1:N_sl - 1
    if k == 1
        gm, _ = generate_gyro_measurements(reshape(zeros(3),3,1), reshape(b0_sl,3,1), dt)
        gyro_rg .= gm[:,1]
    end

    q_mk  = mk_rg[1:4, k];  b_mk  = mk_rg[5:7, k]
    ω_est = gyro_rg - b_mk;  ρ_k   = x_rg[14:16, k]

    # Regulator: point directly at q_sl_final with no intermediate trajectory
    δx_rg    = error_state(q_mk, q_sl_final, ω_est, ρ_k)
    u_rg[:,k] = -K_sl * δx_rg

    gk = copy(gyro_rg)
    x_rg[:,k+1] = RK4_integrator_fullsim_wtime(
        orbit_attitude_dynamics_v2, x_rg[:,k], u_rg[:,k], dt, t_nom_sl[k])

    nbm, nim, gyro_rg, _, _, _, _, bias_rg[:,k+1] =
        get_measurement(x_rg[:,k+1], [epc0 + t_nom_sl[k+1]], bias_rg[:,k])

    mk_rg[:,k+1], Pk_rg[:,:,k+1], _, _ = mekf_step(
        mk_rg[:,k], gk, Pk_rg[:,:,k], nbm, nim,
        dt, V_m_sl, W_m_sl, ss_sl, ms_sl, Mg_sl)
end

# ---- Part 4 Plots ----

# Versine angle profile
θ_prof = [versine_θ(t, θ_sl_tot, T_sl) * 180/π for t in t_nom_sl]
plot(t_nom_sl, θ_prof, linewidth=3, label=false,
     title="Versine Angle Profile (180° Slew)", xlabel="Time (s)", ylabel="θ(t) [deg]")
#savefig("figures/hw4/p4_versine_profile.png")

# Nominal angular velocity magnitude
ω_nom_mag = [norm(ω_nom_sl[:,k]) * 180/π for k = 1:N_sl]
plot(t_nom_sl, ω_nom_mag, linewidth=3, label=false,
     title="Nominal Angular Velocity", xlabel="Time (s)", ylabel="||ω|| [deg/s]")
#savefig("figures/hw4/p4_nominal_omega.png")

# Tracking error vs nominal trajectory
track_err = zeros(N_sl)
for i = 1:N_sl
    qe = L(q_nom_sl[:,i])' * x_sl[7:10, i]
    track_err[i] = 2 * norm(H' * qe) * 180/π
end
plot(t_nom_sl, track_err, linewidth=3, label=false,
     title="Slew Tracking Error (vs Nominal)", xlabel="Time (s)", ylabel="Error [deg]")
savefig("figures/hw4/p4_tracking_error.png")

# Attitude error to final attitude: eigen-axis slew vs regulator
err_sl = zeros(N_sl);  err_rg = zeros(N_sl)
for i = 1:N_sl
    qe_sl = L(q_sl_final)' * x_sl[7:10, i]
    qe_rg = L(q_sl_final)' * x_rg[7:10, i]
    err_sl[i] = 2 * norm(H' * qe_sl) * 180/π
    err_rg[i] = 2 * norm(H' * qe_rg) * 180/π
end
plot(t_nom_sl, err_sl, linewidth=3, label="Eigen-axis slew",
     title="180° Slew: Eigen-axis vs Regulator", xlabel="Time (s)", ylabel="Error to q_final [deg]")
plot!(t_nom_sl, err_rg, linewidth=3, label="Regulator only")
savefig("figures/hw4/p4_slew_vs_regulator.png")

# Nominal vs actual angular velocity
ω_act_mag = [norm(x_sl[11:13,k]) * 180/π for k = 1:N_sl]
plot(t_nom_sl, ω_nom_mag, linewidth=3, label="Nominal")
plot!(t_nom_sl, ω_act_mag, linewidth=3, label="Actual",
      title="Angular Velocity: Nominal vs Actual", xlabel="Time (s)", ylabel="||ω|| [deg/s]")
savefig("figures/hw4/p4_omega_nom_actual.png")

# Wheel torque: feedforward vs total (z-axis = eigen-axis)
plot(t_nom_sl[1:N_sl-1], u_nom_sl[3,:], linewidth=3, label="Nominal τ_z (FF)",
     title="Wheel Torque (z-axis, eigen-axis direction)", xlabel="Time (s)", ylabel="Torque [N·m]")
plot!(t_nom_sl[1:N_sl-1], u_sl[3,:], linewidth=3, label="Total τ_z (FF+FB)")
hline!([ 0.042], color=:red, linestyle=:dot, label="Limit ±0.042 Nm")
hline!([-0.042], color=:red, linestyle=:dot, label=false)
savefig("figures/hw4/p4_wheel_torque_z.png")

# Nominal vs actual wheel angular momentum
ρ_nom_m = [norm(ρ_nom_sl[:,k])  for k = 1:N_sl]
ρ_act_m = [norm(x_sl[14:16,k]) for k = 1:N_sl]
plot(t_nom_sl, ρ_nom_m, linewidth=3, label="Nominal")
plot!(t_nom_sl, ρ_act_m, linewidth=3, label="Actual",
      title="Wheel Angular Momentum Magnitude", xlabel="Time (s)", ylabel="||h_rw|| [N·m·s]")
savefig("figures/hw4/p4_wheel_momentum.png")



plot(u_rg[1:3,:]')