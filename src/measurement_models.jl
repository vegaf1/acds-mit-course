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


#sun sensor model 

"""
additive sun sensor model
"""
#over some desired timeframe pass in both the satellite trajectory and sun trajectory at the desired sample rate
#pass in postion trajectory
#time trajectory is an array of epochs
function generate_sun_measurements(satellite_trajectory, time_trajectory)

    N = size(satellite_trajectory)[2]

    sun_measurements = zeros(3, N)

    for i=1:N
        sun_position = SD.sun_position(time_trajectory[i])

        sun_measurements[:,i] = (sun_position - satellite_trajectory[:,i])/norm((sun_position - satellite_trajectory[:,i]))

    end

    return sun_measurements
end


function generate_noisy_sun_measurements(attitude_trajectory, ground_truth_measurments)

    N = size(attitude_trajectory)[2]
    #misalignment matrix
    M = [1 -1.74e-3 1.74e-3; 1.74e-3 1 -1.74e-3; -1.74e-3 1.74e-3 1] 
    
    #constant bias    
    b = ones(3)*(8.72e-4)

    #covariance matrix for the noise
    W = Matrix(1.0*I,3,3)*(7.61e-5)

    noisy_sun_measurements = zeros(3, N)

    #Q function transforms a quaternion into a rotation matrix
    for i=1:N
        
        noisy_sun_measurements[:,i] = M*(Q(attitude_trajectory[:,i])'*ground_truth_measurments[:,i]) + b + sqrt(W)*randn(3)

    end

    return noisy_sun_measurements

end


#over time, get the magnetometer measurement from IGRF 14 from SatelliteToolboxGeomagneticField.jl 
#time trajectory is a set of epochs
#satellite trajectory is in ECI  
#ground truth measurements
#units of nT 
function generate_magnetometer_measurements(time_trajectory, satellite_trajectory)

    N = size(time_trajectory)[1]

    ground_truth_measurements = zeros(3, N)
    
    #get the measurmements along the entire trajectory    
    for i = 1:N 

        # get decimal date
        date = SD.caldate(time_trajectory[i])
        year = date[1]
        day = SD.day_of_year(time_trajectory[i])
        decimal_date = year + day / 365.2425

        #transform from ECI to ECEF 
        #make sure the state is in meters
        state_ecef = SD.sECItoECEF(time_trajectory[i], satellite_trajectory[:,i]*1000)

        #transform ECEF to geodetic. output is lon [1], lat [2], altitude [3]
        state_geodetic = SD.sECEFtoGEOD(state_ecef)

        #use ecef state to get the IGRF 14 measurement 
        #the time is a fractional year. 
        #the output is in north-east down coordinate system (checkout toolbox docs for definition)
        b_measurement_ned = igrf(decimal_date, state_geodetic[3], state_geodetic[2], state_geodetic[1], Val(:geodetic))

        b_measurement_ecef = ned_to_ecef(b_measurement_ned, state_geodetic[2], state_geodetic[1], state_geodetic[3])

        #compute the rotation matrix to convert from ecef to eci 
        R_eci_ecef = SD.rECEFtoECI(time_trajectory[i])

        #ground truth measurements are in the ECI frame
        ground_truth_measurements[:,i] = R_eci_ecef*b_measurement_ecef

    end

    return ground_truth_measurements

end

#generate the magnetometer measurements 
#these are in units of nanoTeslas
#in the body frame
#attitude trajectory is in quaternions
function generate_noisy_magnetometer_measurements(attitude_trajectory, ground_truth_measurements)

    N = size(attitude_trajectory)[2]

    #from spec sheet

    #misalignment matrix
    M = [1 -1.74e-2 1.74e-2; 1.74e-2 1 -1.74e-2; -1.74e-2 1.74e-2 1] 
    
    #constant bias    
    b = ones(3)*100

    #covariance matrix for the noise
    W = Matrix(1.0*I,3,3)*1600

    #these are in the body frame
    noisy_mag_measurements = zeros(3, N)

    #Q function transforms a quaternion into a rotation matrix
    for i = 1:N 

        noisy_mag_measurements[:,i] = M*(Q(attitude_trajectory[:,i])'*ground_truth_measurements[:,i]) + b + sqrt(W)*randn(3) 

    end

    return noisy_mag_measurements

end


#noisy star tracker measurements

function generate_star_tracker_measurement(attitude_trajectory)

    #number of star trackers
    m = 2 

    #4 arcseconds -> radians
    σ_cross = 0.00001939254724438

    #30 arcseconds -> radians
    #these are from the blue canyon nano startracker data sheet
    σ_roll = 0.0001454441043329

    W = [σ_cross^2 0 0; 0 σ_cross^2 0; 0 0 σ_roll^2]

    N = size(attitude_trajectory)[2]

    star_tracker_measurements = zeros(4*m,N)

    for i = 1:N 
        
        #sample random axis angle vector
        ϕ_1 = sqrt(W)*randn(3)
        ϕ_2 = sqrt(W)*randn(3)
        
        #apply it to the true quaternion

        #left multiply true quaternion with delta quaternion
        star_tracker_measurements[1:4,i] = L(attitude_trajectory[:,i])*expq(ϕ_1)
        star_tracker_measurements[5:8,i] = L(attitude_trajectory[:,i])*expq(ϕ_2)

    end 

    return star_tracker_measurements

end

#these are already in the body frame
#need to pass in the timestep because one of uncertainties is in units of sqrt(hr)...
#ensure dt is in hours
function generate_gyro_measurements(ω_true, bias_true, dt)

    #1-sigma for gyro measurements
    #in rad^2 per hr^2 
    W_hr = [6.85e-4 0 0; 0 1.68e-3 0; 0 0 8.80e-4]

    #convert to (rad/s)
    W = W_hr/(3600^2)

    #this is what is used to simulate the true gyro bias
    #1 sigma standard deviation for bias 
    #converted from degrees/sqrt(hr) to radians/sqrt(hr) 
    b = [0.002268928, 0.002268928, 0.003316126]

    #dt is passed in as an hour already 
    b_rad_hr = b/sqrt(dt)

    #convert rad/hr into rad/s 

    b_rad_s = b_rad_hr*(1/3600)

    #these are in radians per second
    W_b = [b_rad_s[1]^2 0 0; 0 b_rad_s[2]^2 0; 0 0 b_rad_s[3]^2]

    #misalignment matrix (in radians)
    M = [1 -4.36e-3 4.36e-3; 4.36e-3 1 -4.36e-3; -4.36e-3 4.36e-3 1] 

    N = size(ω_true)

    gyro_measurements = zeros(3, N)


    for i=1:N

        gyro_measurements[:,i] = ω_true[:,i] + bias_true[:,i] + randn(3)*cholesky(W)

    end

    return gyro_measurements

end

