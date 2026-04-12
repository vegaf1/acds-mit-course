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

plot(noisy_mag_measurements[1,:])

plot(noisy_mag_measurements[2,:])


plot(noisy_mag_measurements[3,:])
#generate star tracker measurements 
star_tracker_measurements = generate_star_tracker_measurement(attitude_trajectory)

#simulate a random walk for the gyro bias 
true_bias = generate_bias(dt, N)

#simulate gryo measurements 
noisy_gyro_measurements = generate_gyro_measurements(angular_vel_trajectory, true_bias, dt)

#concatenate all the measurements 
noisy_body_measurements = [noisy_sun_measurements; noisy_mag_measurements; star_tracker_measurements]

#ground truth inertial measurements 
inertial_measurements = [sun_eci_measurements; mag_eci_measurements]

#run MEKF 

#initialize states and covariances 

#measurement noise. built to be consistent with the noise when generating the measurements
#W_mekf = BlockDiagonal([Matrix(1.0*I,3,3)*1e-4, Matrix(1.0*I,3,3)*2000, Matrix(1.0*I,3,3)*1e-6, Matrix(1.0*I,3,3)*1e-6])

#measurement noise  
W_mekf = BlockDiagonal([Matrix(1.0*I,3,3)*1e-4, Matrix(1.0*I,3,3)*0.0016, Matrix(1.0*I,3,3)*1e-8, Matrix(1.0*I,3,3)*1e-8])

#process noise 
V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*1e-6, Matrix(1.0*I, 3,3)*1e-8])

#test 
#V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*0.01, Matrix(1.0*I, 3,3)*0.01])

#set to larger...
#V_mekf = BlockDiagonal([Matrix(1.0*I, 3,3)*0.1, Matrix(1.0*I, 3,3)*0.1])

#initial estimate 
#add random noise to the ground truth to analyze how far we have to be from 
#the true solution for the filter to converge

#initial delta rotation from the ground truth (make it consistent with V) 
ϕ0 = sqrt(V_mekf[1:3,1:3])*randn(3)

#sample a random small rotation and apply it to the ground truth initial state (make it consistent with V)
q0 = L(attitude_trajectory[:,1])*expq(ϕ0)

#initial bias estimate from the ground truth (make it consistent with V) 
b0 = true_bias[:,1]+sqrt(V_mekf[4:6,4:6])*randn(3)

x0 = [q0; b0] 

#initial covariance
P0 = Matrix(1.0*I, 6,6).*[ones(3)*1e-3; ones(3)*1e-4]

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
                                                    dt, V_mekf, W_mekf, sun_sensor_specs, mag_sensor_specs)

end

true_bias  
mekf_state 
mekf_state[:,10] 
#analysis

#take the difference between the estimated quaternion and the ground truth 
#represent it as a 3 parameter axis angle
#check that the filter is consistent using this residual

#compare the state trajectories 
plot(mekf_state[1,:], label="estimated") 
plot!(attitude_trajectory[1,:], label="true")

plot(mekf_state[2,:], label="estimated") 
plot!(attitude_trajectory[2,:], label="true")

plot(mekf_state[3,:], label="estimated") 
plot!(attitude_trajectory[3,:], label="true")

plot(mekf_state[4,:], label="estimated") 
plot!(attitude_trajectory[4,:], label="true")

plot(mekf_state[5,:], label="estimated") 
plot!(true_bias[1,:], label="true") 

plot(mekf_state[6,:], label="estimated") 
plot!(true_bias[2,:], label="true") 

plot(mekf_state[7,:], label="estimated") 

plot!(true_bias[3,:], label="true") 



mekf_P



