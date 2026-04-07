using Pkg
Pkg.activate(".")

using LinearAlgebra 
using SatelliteDynamics 
using Plots 
using Convex
using Clarabel
using BenchmarkTools

include("../src/dynamics.jl") 
include("../src/helper.jl")
include("../src/integrators.jl")
include("../src/spacecraft_model.jl")

#orbital dynamics simulation 
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
dt_orbit = T/(N-1)

#state size 
nx = 6

#number of revs to simulate 
n_revs = 10

#N-1 to not repeat the same starting point every rev 
total_N = (N-1)*n_revs

#create a state trajectory to save the result
x_traj = zeros(6, total_N)

#set the initial condtition 
x_traj[:,1] = eci0_km

#simulate forward with no controls
for k=1:total_N-1

    x_traj[:,k+1] = RK4_integrator_wcontrol(orbit_dynamics, x_traj[:,k], zeros(3), dt_orbit)

end

plot(x_traj[1,:], x_traj[2,:], x_traj[3,:], title = "Orbit Trajectory", label = "position")
xlabel!("X [km]") 
ylabel!("           Y [km]")
zlabel!("Z [km ]")

#savefig("../figures/hw1/orbit_traj.png")


#Eulers equation simulation  
#verify stability about each principle

#timestep for attitude sim
dt_euler = 0.1

#number of timesteps (~3 minutes)
N = 2000

#state size for omega sim 
nx_ω= 3

#state trajectory for the angular rate
ω_traj = zeros(nx_ω,N)

#initial omega magnitude (in rpm)
initial_rpm = 3

#convert magnitude to rad/s
initial_magnitude = initial_rpm * 2*pi/60

#this sim is in the principle frame, therefore the 
#principle axes are just [1, 0, 0]...
#can also be done in the body frame, by the rotation matrix to 
#define the principle frame axes in the body frame

#Define different initial conditions for simulation 
#multiply the magnitude with the vector 
#minor axis
#ω0 = [1, 0, 0]*initial_magnitude + 0.1*randn(3)

#intermediate axis
#ω0 = [0, 1, 0]*initial_magnitude + 0.1*randn(3)

#major axis
ω0 = [0, 0, 1]*initial_magnitude + 0.1*randn(3)

ω_traj[:,1] = ω0
#simulate about each principle axis
for k=1:N-1 

    ω_traj[:,k+1] = RK4_integrator(euler_dynamics, ω_traj[:,k], dt_euler)
    
end

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
#all in the principle frame
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