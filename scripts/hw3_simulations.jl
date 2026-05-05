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
include("../src/static_attitude_determination.jl")
include("../src/mekf.jl")

#forward simulate the satellite in an uncontrolled tumble to obtain the ground truth state trajectory

#state size (quaternion + gyrobias size)
nx = 7

#run the simulation at 10 Hz 
dt = 0.1

#simulate around 3 minutes
N = 2000

#initial epoch 
epc = SD.Epoch("2018-12-20")

#Define the orbit initial condition and discretization
#initial orbit (osculating orbital elements)
oe0  = [SD.R_EARTH + 550e3, 0.0, 90.0, 0, 0, 0]

#period (based on the semimajor axis)
period = SD.orbit_period(oe0[1])

#convert to cartesian (in meters)
eci0 = SD.sOSCtoCART(oe0, use_degrees=true)

#scale to km and km/s 
eci0_km = eci0./1000

#define the orbit trajectory 
#state = [position; velocity; attitude; angular velocity; rotor momentum]
orbit_traj_combined = zeros(16, N) 

#sampling a random angular velocity with a 1-sigma standard deviation of 0.1 deg/s
# ω0 = randn(3)*(0.1 * pi/180)

#higher spin 
ω0 = randn(3)*(3.0 * pi/180)


orbit_traj_combined[:,1] = [eci0_km; [1,0,0,0]; ω0; zeros(3)]

#simulating both the position and attitude at 10 Hz. (3 minutes worth of data)
for k=1:N-1

    #assuming no momentum rate control
    orbit_traj_combined[:,k+1] = RK4_integrator_fullsim(orbit_attitude_dynamics, orbit_traj_combined[:,k], zeros(3), dt)

end

#ground truth 
state_trajectory = orbit_traj_combined[1:3, :]
velocity_trajectory = orbit_traj_combined[4:6,:]
attitude_trajectory = orbit_traj_combined[7:10,:]
angular_vel_trajectory = orbit_traj_combined[11:13,:]

#generate measurments (both ground truth inertial and noisy body measurements)

#generate a time trajectory
epochs = []
push!(epochs,epc) 

for i =2:N
    epc_1 = epochs[i-1] + dt
    push!(epochs, epc_1)

end
 
#generate sun measurements in eci frame
sun_eci_measurements = generate_sun_measurements(state_trajectory, epochs)

#generate noisy sun measurements in body frame 
noisy_sun_measurements, sun_sensor_specs = generate_noisy_sun_measurements(attitude_trajectory, sun_eci_measurements)

#generate the magnetometer measurements in the eci frame 
#convert from nT to microtesla
mag_eci_measurements = generate_magnetometer_measurements(epochs, state_trajectory)/1000

#generate noisy magnetometer measurements in the body frame 
noisy_mag_measurements, mag_sensor_specs= generate_noisy_magnetometer_measurements(attitude_trajectory, mag_eci_measurements)


#plot(noisy_sun_measurements[1,:])
#plot(sun_eci_measurements[2,:])

epochs 

plot(noisy_mag_measurements[3,:])
#generate star tracker measurements 
#save the measurement error used when making the measurements
star_tracker_measurements, W_st = generate_star_tracker_measurement(attitude_trajectory)

#simulate a random walk for the gyro bias 
true_bias = generate_bias(dt, N)

#simulate gryo measurements 
noisy_gyro_measurements, M_gyro = generate_gyro_measurements(angular_vel_trajectory, true_bias, dt)

plot(noisy_gyro_measurements[3,:])

#concatenate all the measurements 
noisy_body_measurements = [noisy_sun_measurements; noisy_mag_measurements; star_tracker_measurements]

#ground truth inertial measurements 
inertial_measurements = [sun_eci_measurements; mag_eci_measurements]

#run MEKF 

plot(noisy_gyro_measurements[3,:])

#initialize states and covariances 

#working
#measurement noise  
#W_mekf = BlockDiagonal([Matrix(1.0*I,3,3)*1e-4, Matrix(1.0*I,3,3)*0.0016, Matrix(1.0*I,3,3)*1e-8, Matrix(1.0*I,3,3)*1e-8])

#process noise 
#V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*1e-6, Matrix(1.0*I, 3,3)*1e-8])

############
#testing (this works) 
#W_mekf = BlockDiagonal([Matrix(1.0*I,3,3)*1e-4, Matrix(1.0*I,3,3)*0.0016, Matrix(1.0*I,3,3)*1e-8, Matrix(1.0*I,3,3)*1e-8])
#process noise #working
#V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*1e-6, Matrix(1.0*I, 3,3)*1e-8])
###########3

#using the actual star tracker measurement noise 
W_mekf = BlockDiagonal([Matrix(1.0*I,3,3)*7.61e-5, Matrix(1.0*I,3,3)*0.0016, W_st, W_st])


V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*1e-6, Matrix(1.0*I, 3,3)*2.58e-8])



#initial estimate 
#add random noise to the ground truth to analyze how far we have to be from 
#the true solution for the filter to converge

#this works
#initial delta rotation from the ground truth (make it consistent with V) 
# ϕ0 = sqrt(V_mekf[1:3,1:3])*randn(3) 
# #sample a random small rotation and apply it to the ground truth initial state (make it consistent with V)
# q0 = L(attitude_trajectory[:,1])*expq(ϕ0)
# #initial bias estimate from the ground truth (make it consistent with V) 
# b0 = true_bias[:,1]+sqrt(V_mekf[4:6,4:6])*randn(3)
 
#initialize farther
#1 and 0.001 worked
#20 works
ϕ0 = (20*pi/180)*randn(3) 
#sample a random small rotation and apply it to the ground truth initial state (make it consistent with V)
q0 = L(attitude_trajectory[:,1])*expq(ϕ0)
#initial bias estimate from the ground truth (make it consistent with V) 1e-5 works
b0 = true_bias[:,1]+(1e-5*randn(3))

true_bias 
x0 = [q0; b0] 

#initial covariance
#this should correspond to the initial deviation 
P0 = Matrix(1.0*I, 6,6).*[abs.(ϕ0); ones(3)*1e-5]

mekf_state = zeros(nx, N)
mekf_P = zeros(6,6, N)

#initialize state and covariance 
mekf_state[:,1] = x0
mekf_P[:,:,1] = P0

innovations = zeros(12,N-1)

#run mekf
for k=1:N-1

#for k=1

    mekf_state[:,k+1], mekf_P[:,:,k+1], innovations[:,k] = mekf_step(mekf_state[:,k], noisy_gyro_measurements[:,k], mekf_P[:,:,k], 
                                                    noisy_body_measurements[:,k+1], inertial_measurements[:,k+1],
                                                    dt, V_mekf, W_mekf, sun_sensor_specs, mag_sensor_specs, M_gyro)

end

#analysis

#take the difference between the estimated quaternion and the ground truth 
#represent it as a 3 parameter axis angle
#check that the filter is consistent using this residual

all_time = range(0, step=dt, length=N) 

#compare the state trajectories 
plot(all_time, mekf_state[1,:], label="estimated", xlabel="Time (s)", ylabel="Quaternion Component Value", title= "q0 Comparison", linewidth=3) 
plot!(all_time, attitude_trajectory[1,:], label="true", linewidth=3)
#savefig("figures/hw3/q0_comparison.png")

plot(all_time, mekf_state[2,:], label="estimated", xlabel="Time (s)", ylabel="Quaternion Component Value", title= "q1 Comparison", linewidth=3) 
plot!(all_time, attitude_trajectory[2,:], label="true", linewidth=3)
#savefig("figures/hw3/q1_comparison.png")

plot(all_time, mekf_state[3,:], label="estimated", xlabel="Time (s)", ylabel="Quaternion Component Value", title= "q2 Comparison", linewidth=3) 
plot!(all_time, attitude_trajectory[3,:], label="true", linewidth=3)
#savefig("figures/hw3/q2comparison.png")

plot(all_time, mekf_state[4,:], label="estimated", xlabel="Time (s)", ylabel="Quaternion Component Value", title= "q3 Comparison", linewidth=3) 
plot!(all_time, attitude_trajectory[4,:], label="true", linewidth=3)

#savefig("figures/hw3/q3_comparison.png")

plot(all_time, mekf_state[5,:], label="estimated", xlabel="Time (s)", ylabel="Bias Value", title= "Gyro bias x Comparison", linewidth=3) 
plot!(all_time, true_bias[1,:], label="true", linewidth=3) 
#savefig("figures/hw3/beta_x_comparison.png")

plot(all_time, mekf_state[6,:], label="estimated", xlabel="Time (s)", ylabel="Bias Value", title= "Gyro bias y Comparison", linewidth=3)
plot!(all_time, true_bias[2,:], label="true", linewidth=3) 
#savefig("figures/hw3/beta_y_comparison.png")

plot(all_time, mekf_state[7,:], label="estimated", xlabel="Time (s)", ylabel="Bias Value", title= "Gyro bias z Comparison", linewidth=3) 
plot!(all_time, true_bias[3,:], label="true", linewidth=3) 
#savefig("figures/hw3/beta_z_comparison.png")

mekf_P 
standard_dev = zeros(6, N)

mekf_P  
for i = 1:N 

    #calculate the 3 sigma bound from the filter
    standard_dev[:,i] = 3*sqrt.(diag(mekf_P[:,:,i]))

end

rotation_residual = zeros(3,N)
bias_residaul = zeros(3,N)

bias_residual = mekf_state[5:7,:] - true_bias

for i =1:N

    rotation_residual[:,i] = H'*L(attitude_trajectory[:,i])'*mekf_state[1:4,i]

end

size(all_time)  
#plot the rotation residual (3 axis parameter) vs the 3 sigma bounds 
plot(all_time[10:end], rotation_residual[1,10:end], label="rotation x residual", xlabel="Time (s)", ylabel = "Delta Rotation (rad)", title="x-rotation residuals", linewidth=3)
plot!(all_time[10:end], standard_dev[1,10:end], label="3σ upper bound", linewidth=3)
plot!(all_time[10:end], -standard_dev[1,10:end], label="3σ lower bound", linewidth=3)
#savefig("figures/hw3/Rx_consistency_test3.png")

plot(all_time[10:end], rotation_residual[2,10:end], label="rotation y residual", xlabel="Time (s)", ylabel = "Delta Rotation (rad)", title="y-rotation residuals", linewidth=3)
plot!(all_time[10:end], standard_dev[2,10:end], label="3σ upper bound", linewidth=3)
plot!(all_time[10:end], -standard_dev[2,10:end], label="3σ lower bound", linewidth=3)
#savefig("figures/hw3/Ry_consistency_test3.png")

plot(all_time[10:end], rotation_residual[3,10:end], label="rotation z residual", xlabel="Time (s)", ylabel = "Delta Rotation (rad)", title="z-rotation residuals", linewidth=3)
plot!(all_time[10:end], standard_dev[3,10:end], label="3σ upper bound", linewidth=3)
plot!(all_time[10:end], -standard_dev[3,10:end], label="3σ lower bound", linewidth=3)
#savefig("figures/hw3/Rz_consistency_test3.png")

#plot the residuals of bias with the 3-sigma bounds to check consistency 
plot(all_time[10:end], bias_residual[1,10:end], label="gyro bias x residual", xlabel="Time (s)", ylabel = "Gyro Bias Residual (rad)", title="Gyro Bias x residuals", linewidth=3)
plot!(all_time[10:end], standard_dev[4,10:end], label="3σ upper bound", linewidth=3)
plot!(all_time[10:end], -standard_dev[4,10:end], label="3σ lower bound", linewidth=3)
#savefig("figures/hw3/gyro_x_consistency_test3.png")

plot(all_time[10:end], bias_residual[1,10:end], label="gyro bias y residual", xlabel="Time (s)", ylabel = "Gyro Bias Residual (rad)", title="Gyro Bias y residuals", linewidth=3)
plot!(all_time[10:end], standard_dev[5,10:end], label="3σ upper bound", linewidth=3)
plot!(all_time[10:end], -standard_dev[5,10:end], label="3σ lower bound", linewidth=3)
#savefig("figures/hw3/gyro_y_consistency_test3.png")

plot(all_time[10:end], bias_residual[1,10:end], label="gyro bias z residual", xlabel="Time (s)", ylabel = "Gyro Bias Residual (rad)", title="Gyro Bias z residuals", linewidth=3)
plot!(all_time[10:end], standard_dev[6,10:end], label="3σ upper bound", linewidth=3)
plot!(all_time[10:end], -standard_dev[6,10:end], label="3σ lower bound", linewidth=3)
#savefig("figures/hw3/gyro_z_consistency_test3.png")

rotation_residual 

bias_residual 
