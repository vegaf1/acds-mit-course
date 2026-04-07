#measurment models

"""
Function for generating noisy bearing measurements 
Input: standard deviation, number of measurements 
Output: A trajectory of bearing measurements, noisy bearing measurements, and the associated covariance
"""
function generate_noisy_bearing_measurements(std_dev, N_measurements)

    #bearing vector measurements in the body frame 
    b_measurements = zeros(3, N_measurements)

    #bearing vector mea
    noisy_b_measurements = zeros(3,N_measurements)

    #convert the degrees standard deviation into radians
    std_dev_rad = deg2rad(std_dev)

    #get the radians
    var_rad = std_dev_rad^2

    all_P = zeros(3,3,N_measurements)

    for i =1:N_measurements

        direction = randn(3)

        b_measurements[:,i] = direction/norm(direction)

        #last term is for numerical reasons so that we can take the cholesky decomp 
        all_P[:,:,i] = var_rad*(I - b_measurements[:,i]*b_measurements[:,i]') + (1e-15 * Matrix(1.0I, 3,3))

        #sample noise
        #direction is now the mean 
        noisy_b_measurements[:,i] = b_measurements[:,i] + cholesky(all_P[:,:,i]).L*randn(3)

        noisy_b_measurements[:,i] = noisy_b_measurements[:,i]/norm(noisy_b_measurements[:,i])

    end

    return b_measurements, noisy_b_measurements, all_P

end

"""
Function to create inertial measurements 
Input: True attitude and body measurements (no noise)
Output: Ground truth inertial measurements

"""
function create_inertial_measurements(Q_true, b_m, N_measurements)

    i_m = zeros(3, N_measurements)

    for i=1:N_measurements

        i_m[:,i] = Q_true*b_m[:,i]

    end

    return i_m

end