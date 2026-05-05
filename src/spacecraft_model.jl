#Define the spacecraft model

#length of each body in their own coordinate system 
#spacecraft decomposed into 3 pieces
#see Figure 2 from report 
#used for moment of inertia calculation 

#prism 1 (in cm)
lx_1 = 10
ly_1 = 20
lz_1 = 34

#prism 2 (in cm)
lx_2 = 1
ly_2 = 100
lz_2 = 34 

#prism 3 (in cm)
lx_3 = 1
ly_3 = 100
lz_3 = 34

#mass of each rectangular prism (in kg)
m1 = 12
m2 = 1
m3 = 1

#distance from the body frame origin to each prism COM (in cm) 
r1 = [0,0,0]
r2 = [35.36, 35.36, 0]
r3 = [-35.36, -35.36, 0]

#center of mass
r_com = (m1*r1 + m2*r2 + m3*r3)/(m1 + m2 + m3)

#individual moment of inertias (formula for the rectangular prism)
#source: https://dynref.engr.illinois.edu/rem.html
J_1 = Matrix(1.0*I,3,3).*[(1/12)*m1*(ly_1^2 + lz_1^2); (1/12)*m1*(lz_1^2 + lx_1^2); (1/12)*m1*(lx_1^2 + ly_1^2)]
J_2 = Matrix(1.0*I,3,3).*[(1/12)*m2*(ly_2^2 + lz_2^2); (1/12)*m2*(lz_2^2 + lx_2^2); (1/12)*m2*(lx_2^2 + ly_2^2)]
J_3 = Matrix(1.0*I,3,3).*[(1/12)*m3*(ly_3^2 + lz_3^2); (1/12)*m3*(lz_3^2 + lx_3^2); (1/12)*m3*(lx_3^2 + ly_3^2)]

#combined inertia calculation 
#uses the parallel axis theorem 
J_b = J_1 + J_2 + J_3 + m1*(r1'*r1*I - r1*r1') + m2*(r2'*r2*I - r2*r2') + m3*(r3'*r3*I - r3*r3')

#principle axes is the bases where the body frame moment of inertia is diagonal 
#take an eigen decomp to find this 
J_b_eig = eigen(J_b)

#this is a rotation that can convert from principle axes to body frame
#this is the set of principle axes in the body frame
principle_axes = J_b_eig.vectors 

#moment of inertia about principle axes
J_principle = principle_axes'*J_b*principle_axes 

#get both principle and body inertia into units of meters and seconds
J_principle_SI = J_principle/(100^2)

J_b_SI = J_b/(100^2)

#define solar panel normal (used as a spin axis in hw2)
solar_normal = [-sqrt(2)/2; sqrt(2)/2; 0]

"""
Perturbs the inertia matrix by some small factor. Used to ensure simulation axis 
is not parallel to a principle axis

# Arguments: Inertia Matrix 

# Return: Perturbed Inertia Matrix 
"""
function perturb_inertia(J)

    #sample random perturbations for eigen values and vectors 
    d = 0.1*randn(3)
    v = 0.1*randn(3)

    J_eig = eigen(J)
    D = J_eig.values.*Matrix(1.0*I,3,3)
    V = J_eig.vectors
    D_tilde = D*(Matrix(1.0*I,3,3) + diagm(d))
    V_tilde = V*exp(hat(v))

    J_tilde = V_tilde*D_tilde*V_tilde'

    return J_tilde 

end

#define a pertubed body inertia
J_b_SI_perturbed = perturb_inertia(J_b_SI) 


function actuator_jacobians()

    #positions of the thrusters r (in cm)
    r1 = [10, 5, -17]
    r2 =  [-10,5, -17]
    r3 =  [10, -5, -17]
    r4 =  [-10, -5, -17]

    a1 = [0, -sind(20), -cosd(20)]
    a2 = [0, -sind(20), -cosd(20)]
    a3 = [0, sind(20), -cosd(20)]
    a4 = [0, sind(20), -cosd(20)]

    #divided by 100 to get into meters
    #thruster jacobian
    Bt = [cross(r1/100, a1) cross(r2/100, a2) cross(r3/100, a3) cross(r4/100, a4)]

    #we assume that the wheels are along the body axis. Therefore the wheel jacobian is identity matrix
    Bw = Matrix(1.0*I,3,3)

    return Bt, Bw

end
#may be useful to define a struct for the terms that are used the most...

# Face geometry: centroids [m], outward normals, and areas [m²] in the body frame
# CoM is at the body frame origin, so centroids are the moment arms directly.
# 6 bus faces + 5 solar panel 1 faces + 5 solar panel 2 faces = 16 total

const face_centroids = [
    # Bus
    [0.0,    0.05,  0.0 ],
    [0.0,   -0.05,  0.0 ],
    [0.10,   0.0,   0.0 ],
    [-0.10,  0.0,   0.0 ],
    [0.0,    0.0,   0.17],
    [0.0,    0.0,  -0.17],
    # Solar Panel 1
    [0.3536,  0.3536,  0.0 ],
    [0.3536,  0.3536,  0.0 ],
    [0.3536,  0.3536,  0.17],
    [0.3536,  0.3536, -0.17],
    [0.7071,  0.7071,  0.0 ],
    # Solar Panel 2
    [-0.3536, -0.3536,  0.0 ],
    [-0.3536, -0.3536,  0.0 ],
    [-0.3536, -0.3536,  0.17],
    [-0.3536, -0.3536, -0.17],
    [-0.7071, -0.7071,  0.0 ],
]

const face_normals = [
    # Bus
    [0.0,  1.0,  0.0],
    [0.0, -1.0,  0.0],
    [1.0,  0.0,  0.0],
    [-1.0, 0.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0,  0.0, -1.0],
    # Solar Panel 1
    [-sqrt(2)/2,  sqrt(2)/2, 0.0],
    [ sqrt(2)/2, -sqrt(2)/2, 0.0],
    [0.0,  0.0,  1.0],
    [0.0,  0.0, -1.0],
    [ sqrt(2)/2,  sqrt(2)/2, 0.0],
    # Solar Panel 2
    [-sqrt(2)/2,  sqrt(2)/2, 0.0],
    [ sqrt(2)/2, -sqrt(2)/2, 0.0],
    [0.0,  0.0,  1.0],
    [0.0,  0.0, -1.0],
    [-sqrt(2)/2, -sqrt(2)/2, 0.0],
]

const face_areas = [
    # Bus [m²]
    0.0680, 0.0680, 0.0340, 0.0340, 0.0200, 0.0200,
    # Solar Panel 1 [m²]
    0.2924, 0.2924, 0.0086, 0.0086, 0.0086,
    # Solar Panel 2 [m²]
    0.2924, 0.2924, 0.0086, 0.0086, 0.0086,
]
