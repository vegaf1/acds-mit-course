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


"""
Drag torque in the body frame [N·m]. (from wertz book)
Uses a flat-plate model summed over all spacecraft faces.\
v_eci [km/s], r_eci [km], ρ_atm [kg/m³]
q [attitude]

"""
function drag_torque(x, ρ_atm, cd)

    ω_E = 7.2921e-5  # Earth rotation rate [rad/s]

    r_eci = x[1:3]
    v_eci = x[4:6]
    q = x[7:10]
    ω_body = x[11:13]

    v_ms   = v_eci * 1000.0
    r_m    = r_eci * 1000.0

    #find the equation of the velocity relative to the atmosphere (in ECI)
    #from the montebruck satellite orbits book
    v_rel  = v_ms - cross([0.0, 0.0, ω_E], r_m)

    #angular velocity relative to the atmosphere
    ω_rel =  ω_body - Q(q)' * [0, 0, ω_E]

    τ = zeros(3)

    for i in eachindex(face_normals)
        #translational velocity
        #from wertz book
        v_rel_body = (Q(q)' * v_rel) + cross(ω_rel, face_centroids[i])
        v_rel_body_norm = v_rel_body/norm(v_rel_body)

        #check all the faces that are against the velocity
        nv_condition = dot(face_normals[i], v_rel_body_norm)
        
        #If it is true, apply an opposing drag force 
        if nv_condition > 0
            F_i = -0.5* cd *ρ_atm * ((v_rel_body'*v_rel_body))* nv_condition* v_rel_body_norm*face_areas[i]
            τ  += cross(face_centroids[i], F_i)
        end
    end
    return τ

end

"""
calculate the gravity gradient torque
source: acds course notes
"""
function gravity_gradient_torque(x)

    #ECI position in meters
    r_m = x[1:3]*1000

    q = x[7:10]

    #rotate ECI position into body frame
    r_b = Q(q)' * r_m

    #gravity gradient torque [N·m] — all quantities in body frame
    τ_gg = cross(((3*SD.GM_EARTH)/(r_b'*r_b)^(5/2))*r_b, J_b_SI_perturbed*r_b)

    return τ_gg

end


"""
#function to simulate position and attitude (higher fidelity)
Orbit Dynamics + gyrostat model
Includes drag acceleration, drag torque, and gravity gradient torque
State: [position, velocity, attitude quaternion, angular velocity, rotor momentum]
Control: [wheel momenta (3); thruster force(4)]
Time in seconds
"""
function orbit_attitude_dynamics_v2(x, u, time_s)

    #position (km)
    r = x[1:3]

    #position (m)
    r_m = r*1000

    #velocity
    v = x[4:6]

    #velocity [m/s]
    v_m = v*1000

    #attitude quaternion
    q = x[7:10]

    #angular velocity
    ω = x[11:13]

    #rotor momentum in body frame  
    h_rw = x[14:16]

    #momentum rate in the body frame using actuator jacobian
    h_rw_dot = Bw*u[1:3]

    #control is thruster forces
    #convert to torque in body frame with thruster jacobian
    thruster_torques = Bt*u[4:7]

    epc = epc0 + time_s

    PN = SD.bias_precession_nutation(epc)
    Earth_r = SD.earth_rotation(epc)
    rpm  = SD.polar_motion(epc)

    R = rpm*Earth_r*PN

    #density from harris priester model
    ρ_atm = SD.density_harris_priester(epc,r_m)

    cd = 2.0 #drag coefficient (assumption)

    #bus mass 12 kg, assuming 1kg for each panel
    m = 14.0

    #drag acceleration (attitude-independent scalar area)
    area_drag = 0.36 #in m2

    #need to convert a_drag back to km/s2 at the end
    a_drag = SD.accel_drag([r_m; v_m], ρ_atm, m, area_drag, cd, Array{Real,2}(PN))/1000

    #gravity gradient torque [N·m]
    τ_gg = gravity_gradient_torque(x)

    #aerodynamic drag torque [N·m]
    τ_drag = drag_torque(x, ρ_atm, cd)

    env_body_torques = τ_gg + τ_drag

    #with the perturbed inertia (body frame) with environmental disturbances
    ω_dot = -J_b_SI_perturbed\(h_rw_dot + hat(ω)*(J_b_SI_perturbed*ω + h_rw) - env_body_torques - thruster_torques)

    #ω_dot = -J_b_SI_perturbed\(h_rw_dot + hat(ω)*(J_b_SI_perturbed*ω + h_rw))

    q_dot = 0.5*G(q)*ω

    #earth gravitational constant [km3/s2]
    μ = SD.GM_EARTH/(1000^3)

    #radius of the Earth in km
    r_earth = SD.R_EARTH/1000

    a_grav = (-μ/norm(r)^3)*r

    a_J2 = (-3/2)*((SD.J2_EARTH*μ*r_earth^2)/(norm(r)^5))*
    [(1-5*(r[3]/norm(r))^2)*r[1]; (1-5*(r[3]/norm(r))^2)*r[2]; (3-5*(r[3]/norm(r))^2)*r[3]]

    #total acceleration (in km/s2)
    a = a_grav + a_J2 + a_drag

    x_dot = [v; a; q_dot; ω_dot; h_rw_dot]

    return x_dot

end

