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

    #use log here instaed (this is for small errors)
    #ϕ = 2*H'*L(q_d)'*qk

    ϕ = 2*logq(L(q_d)'*qk)

    return [ϕ; ω; ρ]

end


function linear_attitude_dynamics_tvlqr(ω, J, ρ, dt_v)


    A = [-hat(ω) Matrix(1.0*I,3,3) zeros(3,3); 
         zeros(3,3) -inv(J)*(hat(ω)*J - hat(J*ω+Bw*ρ)) -inv(J)*hat(ω)*Bw; 
         zeros(3,9)
         ]

    B = [zeros(3,7); 
        -inv(J)*Bw zeros(3,4);
        Bw zeros(3,4)]


    #use the exponential map to get the discrete time matrices
    C = exp([A B; zeros(7,9) zeros(7,7)]*dt_v)

    A_d = C[1:9, 1:9]
    B_d = C[1:9, 10:16]

    return A_d, B_d

end

"""
Compute the cost to go and linear feedback gain matrices for tvlqr
Qn - terminal cost 
Q - state cost 
R - control cost 
N - number to timesteps 
orbit_dt- timestep (s)
"""
function linear_attitude_dynamics_tvlqr(Qn, Q, R, N, dt_v, ω_traj, J, ρ_traj)

    #cost to go matrices 
    P = zeros(9,9, N)

    #feedback gains 
    K = zeros(7,9, N-1)

    A = zeros(9,9,N-1)
    
    B = zeros(9,7, N-1)
    
    P[:,:,N] = Qn

    #ricatti recursion
    for k = (N-1):-1:1

        A[:,:,k], B[:,:,k] = linear_attitude_dynamics_tvlqr(ω_traj[:,k], J, ρ_traj[:,k],dt_v)
        K[:,:,k] .= (R + B[:,:,k]'*P[:,:,k+1]*B[:,:,k])\(B[:,:,k]'*P[:,:,k+1]*A[:,:,k])
        P[:,:,k] .= Q + A[:,:,k]'*P[:,:,k+1]*(A[:,:,k]-B[:,:,k]*K[:,:,k])

    end

    return P, K 

end
