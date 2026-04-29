"""
Linear system for 
LQR
x - axis angle ϕ, angular velocity ω
u - rotor momentum time derivative ρ_dot (rotor torque), body torques τ
return the discrete dynamics model 
"""

function linear_attitude_dynamics(orbit_dt)

    #continous dynamics model 
    A = [zeros(3,3) Matrix(1.0*I, 3,3) zeros(3,3); zeros(3,3) zeros(3,3) zeros(3,3); zeros(3,3) zeros(3,3) zeros(3,3)]
    B = [zeros(3,3) zeros(3,4); -inv(J_b_SI_perturbed)*Bw inv(J_b_SI_perturbed)*Bt; Bw zeros(3,4)]

    #use the exponential map to get the discrete time matrices
    C = exp([A B; zeros(7,9) zeros(7,7)]*orbit_dt)

    A_d = C[1:9, 1:9]
    B_d = C[1:9, 10:16]

    return A_d, B_d

end

"""
Compute the cost to go and linear feedback gain matrices
Qn - terminal cost 
Q - state cost 
R - control cost 
N - number to timesteps 
orbit_dt- timestep (s)
"""
function ricatti_recursion(Qn, Q, R, N, orbit_dt)

    #discrete dynamics
    A, B = linear_attitude_dynamics(orbit_dt)

    #cost to go matrices 
    P = zeros(9,9, N)

    #feedback gains 
    K = zeros(7,9, N-1)
    
    P[:,:,N] = Qn

    #ricatti recursion
    for k = (N-1):-1:1

        K[:,:,k] .= (R + B'*P[:,:,k+1]*B)\(B'*P[:,:,k+1]*A)
        P[:,:,k] .= Q + A'*P[:,:,k+1]*(A-B*K[:,:,k])

    end

    return P, K 

end

"""
Calculate error state
q - current quaternion 
qd - desired quaternion 
ω - current angular velocity
return axis angle; linearized ω
x - xgoal for quaternions
"""

function error_state(qk, q_d, ω, ρ)

    # shortest path: if on wrong hemisphere, flip sign so LQR corrects in the right direction
    if dot(qk, q_d) < 0
        qk = -qk
    end

    ϕ = 2*H'*L(q_d)'*qk

    return [ϕ; ω; ρ]

end
