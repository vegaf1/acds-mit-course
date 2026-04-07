#static attitude determination methods 

"""
Solve Wabha's problem as a convex semidefinite program
Input: n_b_m is a set of noisy body measurements
       i_m are the measurements in the intertial frame
       N is the number of measurements 
Output: Rotation matrix between body and inertial 
"""
function solve_wabha_sdp(n_b_m, i_m, N_measurements)

    #create the attitude profile matrix 
    w = ones(N_measurements)

    B = zeros(3,3)

    for k = 1:N_measurements 

        B += w[k]*n_b_m[:,k]*i_m[:,k]'

    end

    Q = Variable(3,3)
    prob = maximize(tr(B*Q), [Matrix(1.0*I, 3,3) Q'; Q Matrix(I, 3,3)] ⪰ 0)

    solve!(prob, Clarabel.Optimizer)

    return Q.value

end

"""
Solve Wabha's problem using an SVD
Input: n_b_m is a set of noisy body measurements
       i_m are the measurements in the intertial frame
       N is the number of measurements 
Output: Rotation matrix between body and inertial 
"""
function solve_wabha_svd(n_b_m, i_m, N_measurements)

    #create the attitude profile matrix 
    w = ones(N_measurements)

    B = zeros(3,3)

    for k = 1:N_measurements 

        B += w[k]*n_b_m[:,k]*i_m[:,k]'

    end

    F = svd(B)

    Q = F.V*F.U'

    return Q 

end

"""
Function to evalute the accuracy between an estimated 
rotation and the ground truth rotation 
Input: Ground truth rotation, estimated rotation 
Output: Error in degrees between both
"""
function evaluate_accuracy(Q_true, Q_estimate)

    error_deg = (180/pi)*norm(unhat(log(Q_estimate'*Q_true)))

    return error_deg
end
