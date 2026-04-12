#sequential attitude estimator 
#will put omega in the control

"""
quaternion forward propagation assuming 
constant angular velocity between timestep 
"""
function state_prediction(x, u, dt)

    #quaternion
    q = x[1:4]

    #gyro bias
    β = x[5:7]

    #angular velocity 
    ω = u 

    #delta quaternion
    Δq = expq(0.5*dt*(ω - β))

    #apply delta quaternion to current state
    q1 = L(q)*Δq

    #identity dynamics
    β1 = β

    x1 = [q1; β1]

    return x1

end

"""
dynamics jacobian
"""
function state_prediction_deriv(x, u, dt)

    q = x[1:4]
    β = x[5:7]
    ω = u 

    Δq = expq(0.5*dt*(ω - β))
    q1 = L(q)*Δq

    ∂ϕk1_∂ϕk = G(q1)'*R(Δq)*G(q)
    ∂ϕk1_∂βk = -0.5*dt*G(q1)'*G(q)
    ∂βk1_∂ϕk = zeros(3,3)
    ∂βk1_∂βk = Matrix(1.0*I, 3,3)

    A = [∂ϕk1_∂ϕk ∂ϕk1_∂βk; ∂βk1_∂ϕk ∂βk1_∂βk]

    return A 

end

"""
difference between true measurements (noisy) and predicted measurements (based on current state)
x - state - [quaternion [1:4], gyro bias [5:7]]
yb[1:3] - sun sensor body measurement
yb[4:6] - mag body measurment
yb[7:10] - star tracker 1 measurement
yb[11:14] - star tracker 2 measurement 

yi[1:3] - sun sensor inertial measurement
yi[4:6] - mag inertial measurement

sun_sensor specs[1]: 
M_s - misalignment matrices sun sensor 

sun_sensor_specs[2]
b_s - bias vector for sun sensor 

mag_sensor specs[1]: 
M_b - misalignment matrices magnetomter 

mag_sensor_specs[2]
m_b - bias vector for magnetomter

#put all these in a struct to pass them cleanly.

#for star trackers - innovation is 3 parameter per star tracker

"""
function innovation(x,yb, yi, sun_sensor_specs, mag_sensor_specs) 

    z = zeros(12)

    #estimated quaternion 
    q = x[1:4]

    #sun sensor misalignment matrix 
    M_s = sun_sensor_specs[1]

    #sun sensor bias
    b_s = sun_sensor_specs[2]

    #magnetometer misalignment matrix 
    M_b = mag_sensor_specs[1]

    #magnetomter bias 
    b_m = mag_sensor_specs[2]

    #sun sensor difference 
    # z[1:3] = yb[1:3] - (M_s*(Q(q)'*yi[1:3]) + b_s)

    # #magnetomter difference 
    # z[4:6] = yb[4:6] - (M_b*(Q(q)'*yi[4:6]) + b_m) 

    #order changed because there is a negative sign in the Kalman gain 
    #sun sensor difference 
    z[1:3] =  (M_s*(Q(q)'*yi[1:3]) + b_s) - yb[1:3]

    #magnetomter difference 
    z[4:6] = (M_b*(Q(q)'*yi[4:6]) + b_m) - yb[4:6] 

    #this is from the notes 
    #star tracker 1 difference
    z[7:9] = H'*L(q)'*yb[7:10]

    #star tracker 2 difference 
    z[10:12] = H'*L(q)'*yb[11:14]

    #z[7:9] = G(q)'*yb[7:10]

    #star tracker 2 difference 
    #z[10:12] = G(q)'*yb[11:14]

    return z 

end

"""
innovation jacobian
Inputs: 
yb[1:3] - sun sensor body measurement
yb[4:6] - mag body measurment
yb[7:10] - star tracker 1 measurement
yb[11:14] - star tracker 2 measurement 

yi[1:3] - sun sensor inertial measurement
yi[4:6] - mag inertial measurement

sun_sensor specs[1]: 
M_s - misalignment matrices sun sensor 

sun_sensor_specs[2]
b_s - bias vector for sun sensor 

mag_sensor specs[1]: 
M_b - misalignment matrices magnetomter 

mag_sensor_specs[2]
m_b - bias vector for magnetomter
"""
function innovation_deriv(x,yb, yi, sun_sensor_specs, mag_sensor_specs)

    #quaternion 
    q = x[1:4]

    sun_inertial = yi[1:3]
    mag_inertial = yi[4:6]

    M_s = sun_sensor_specs[1]
    M_b = mag_sensor_specs[1]

    #quaternion estimate from star tracker 1
    qst1 = yb[7:10]

    #quaternion estimate from star tracker 2 
    qst2 = yb[11:14] 

    #rows - number of elements in the innovation vector (3 from sun measurement, 3 from magnetometer measurement, 3 from star tracker 1, 3 from startracker 2)
    #columns - 3 parameter attitude representation (ϕ) + 3 parameter gyro bias (b)
    innovation_deriv = zeros(12, 6)

    #deriv of innovation (sun) wrt quaternion turned into 3 parameter representation with attitude jacobian (∂q/∂ϕ))
    #G is the attitude jacobian 
    innovation_deriv[1:3, 1:3] = M_s*(H'*(L(q)'*L(H*sun_inertial) + R(q)*R(H*sun_inertial)*T)*G(q))

    #deriv of innovation (magnetometer) wrt quaternion turned into 3 parameter representation with attitude jacobian
    innovation_deriv[4:6, 1:3] = M_b*(H'*(L(q)'*L(H*mag_inertial) + R(q)*R(H*mag_inertial)*T)*G(q))

    #deriv of innovation (star tracker 1) wrt quaternion 
    innovation_deriv[7:9, 1:3] = H'*R(qst1)*T*G(q)

    #deriv of innotvation (star tracker 2) wrt quaternion
    innovation_deriv[10:12, 1:3] = H'*R(qst2)*T*G(q)

    return innovation_deriv

end


"""
Function to take a step in the filter
Input: xk - state at timestep k 
       yk - current gyro measurement (timestep k)
       Pk - current state covariance (timestep k)
       ybk1 - noisy body measurement at timestep k+1
       yik1 - ground truth inertial measurement at timestep k+1
       dt - timestep 
       V - process covariance 
       W - measurement covariance
       sun_sensor_specs - sun sensor misalignment matrix and bias 
       mag_sensor_specs - mag sensor misalignment matrix and bias 
"""
function mekf_step(xk, uk, Pk, ybk1, yik1, dt, V, W, sun_sensor_specs, mag_sensor_specs)

    #Prediction step 
    x_prediction = state_prediction(xk, uk, dt)
    A = state_prediction_deriv(xk, uk, dt)
    P_prediction = A*Pk*A' + V

    #Innovation step 
    z = innovation(x_prediction, ybk1, yik1, sun_sensor_specs, mag_sensor_specs)
    C = innovation_deriv(x_prediction, ybk1, yik1, sun_sensor_specs, mag_sensor_specs)

    #println("this is C: ", C)

    S = C*P_prediction*C' + W 

    #Kalman Gain 
    K = P_prediction*C'/S

    #K = P_prediction*C'*inv(S) 

    #println("this is the kalman gain: ", K)

    #Update
    #delta state is only size 6 
    #since there is a negative here, the innovation on the vector measurements needs to be flipped 
    delta_x= -K*z

    #println("delta x: ", delta_x)

    Δϕ = delta_x[1:3] 
    Δβ= delta_x[4:6] 
    
    #transform Δϕ to Δq
    Δq = [sqrt(1 - Δϕ'*Δϕ); Δϕ]

    #corrected state 
    xk1 = zeros(size(xk))

    #updated quaternion 
    xk1[1:4] = L(x_prediction[1:4])*Δq

    #renormalize 
    xk1[1:4] = xk1[1:4]/norm(xk1[1:4])

    #updated bias 
    xk1[5:7] = x_prediction[5:7] + Δβ

    #print("this is delta β", Δβ)

    Pk1 = (Matrix(1.0*I, 6,6) - K*C)*P_prediction*(Matrix(1.0*I, 6,6) - K*C)' + K*W*K'

    return xk1, Pk1, z

end
