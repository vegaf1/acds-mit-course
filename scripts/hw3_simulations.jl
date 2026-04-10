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

include("../src/dynamics.jl") 
include("../src/helper.jl")
include("../src/integrators.jl")
include("../src/spacecraft_model.jl")
include("../src/measurement_models.jl")
include("../src/static_attitude_determination.jl")

#forward simulate the satellite in an uncontrolled tumble to obtain the ground truth state trajectory

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
T = SD.orbit_period(oe0[1])

#convert to cartesian (in meters)
eci0 = SD.sOSCtoCART(oe0, use_degrees=true)

#scale to km and km/s 
eci0_km = eci0./1000

#define the orbit trajectory 
#state = [position; velocity attitude; angular velocity; rotor momentum]
orbit_traj_combined = zeros(16, N) 

#sampling a random angular velocity with a 1-sigma standard deviation of 0.1 deg/s
ω0 = randn(3)*(0.1 * pi/180)

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
#generate measurments (both ground truth inertial and noisy body measurements)

#generate a time trajectory
epochs = []
push!(epochs,epc) 

for i =2:N
    epc_1 = epochs[i-1] + dt
    push!(epochs, epc_1)

end

state_trajectory 

#generate sun measurements in eci frame
sun_eci_measurements = generate_sun_measurements(state_trajectory, epochs)

#generate noisy sun measurements in body frame 
noisy_sun_measurements = generate_noisy_sun_measurements(attitude_trajectory, sun_eci_measurements)

#generate the magnetometer measurements in the eci frame 
mag_eci_measurements = generate_magnetometer_measurements(epochs, state_trajectory)

#generate noisy magnetometer measurements in the body frame 
noisy_mag_measurements= generate_noisy_magnetometer_measurements(attitude_trajectory, mag_eci_measurements)

#generate star tracker measurements 
star_tracker_measurements = generate_star_tracker_measurement(attitude_trajectory)

#run MEKF 

