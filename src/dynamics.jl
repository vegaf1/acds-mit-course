#dynamics models

"""
#define continous orbit dynamics function
input: state x - [position, velocity], units [km, s]
control: additive acceleration
current model has gravitation acceleration and J2
"""
function orbit_dynamics(x, u)

    #position
    r = x[1:3]

    #velocity 
    v = x[4:6]

    #earth gravitational constant [km3/s2]
    μ = SD.GM_EARTH/(1000^3)
    
    #radius of the Earth in km
    r_earth = SD.R_EARTH/1000

    a_grav = (-μ/norm(r)^3)*r 

    a_J2 = (-3/2)*((SD.J2_EARTH*μ*r_earth^2)/(norm(r)^5))*
    [(1-5*(r[3]/norm(r))^2)*r[1]; (1-5*(r[3]/norm(r))^2)*r[2]; (3-5*(r[3]/norm(r))^2)*r[3]]

    #total acceleration
    a = a_grav + a_J2 + u

    x_dot = [v; a]

end


"""
#function to simulate position and attitude
Orbit Dynamics + gyrostat model
State: [position, velocity, attitude quaternion, angular velocity, rotor momentum] 
Control: [rotor momentum rate]
"""
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

    #with the perturbed inertia 
    ω_dot = -J_b_SI_perturbed\(ρ_dot + hat(ω)*(J_b_SI_perturbed*ω + ρ))

    q_dot = 0.5*G(q)*ω

    #earth gravitational constant [km3/s2]
    μ = SD.GM_EARTH/(1000^3)
    
    #radius of the Earth in km
    r_earth = SD.R_EARTH/1000

    a_grav = (-μ/norm(r)^3)*r 

    a_J2 = (-3/2)*((SD.J2_EARTH*μ*r_earth^2)/(norm(r)^5))*
    [(1-5*(r[3]/norm(r))^2)*r[1]; (1-5*(r[3]/norm(r))^2)*r[2]; (3-5*(r[3]/norm(r))^2)*r[3]]

    #total acceleration
    a = a_grav + a_J2

    x_dot = [v; a ;q_dot; ω_dot; ρ_dot]

    return x_dot

end

"""
Euler's equation in the principle frame 
Input: angular velocity 
no control input
"""
function euler_dynamics(ω)

    ω_dot = zeros(3)

    ω_dot[1] = (-(J_principle_SI[3,3] - J_principle_SI[2,2])*ω[2]*ω[3])/J_principle_SI[1,1] 
    ω_dot[2] = (-(J_principle_SI[1,1] - J_principle_SI[3,3])*ω[1]*ω[3])/J_principle_SI[2,2] 
    ω_dot[3] = (-(J_principle_SI[2,2] - J_principle_SI[1,1])*ω[1]*ω[2])/J_principle_SI[3,3] 

    return ω_dot 

end

"""
Euler equation in the body frame 
x - [angular velocity; rotor momentum]
u - [rotor momentum rate]
"""
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
