#RK4 integrators and helper functions
"""
rk4 implementation with no control input 
f - dynamics function 
x - state 
dt - timestep 
"""
function RK4_integrator(f, x, dt)

    k1 = f(x)
    k2 = f(x + (dt/2)*k1)
    k3 = f(x + (dt/2)*k2)
    k4 = f(x + (dt)*k3)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return xk_1 

end

"""
RK4 function with a control input 
Input: 
f - continous dynamics function 
x - state 
u - control 
dt - timestep 
"""
function RK4_integrator_wcontrol(f, x, u, dt)

    k1 = f(x, u)
    k2 = f(x + (dt/2)*k1, u)
    k3 = f(x + (dt/2)*k2, u)
    k4 = f(x + (dt)*k3, u)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return xk_1 

end

"""
RK4 function for full state (attitude included)
Input: 
f - continous dynamics function 
x - state 
u - control 
dt - timestep
"""
function RK4_integrator_fullsim(f, x, u, dt)

    k1 = f(x, u)
    k2 = f(x + (dt/2)*k1, u)
    k3 = f(x + (dt/2)*k2, u)
    k4 = f(x + (dt)*k3, u)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    #re normalize the quaternion 
    xk_1[7:10] .= xk_1[7:10]/norm(xk_1[7:10])

    return xk_1 

end

"""
RK4 function for full state (attitude included)
Input: 
f - continous dynamics function 
x - state 
u - control 
dt - timestep
t - elapsed time since the start time (in seconds)
"""
function RK4_integrator_fullsim_wtime(f, x, u, dt, t)

    k1 = f(x, u, t)
    k2 = f(x + (dt/2)*k1, u, t+(dt/2))
    k3 = f(x + (dt/2)*k2, u, t+(dt/2))
    k4 = f(x + (dt)*k3, u, t+dt)

    xk_1 = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    #re normalize the quaternion 
    xk_1[7:10] .= xk_1[7:10]/norm(xk_1[7:10])

    return xk_1 

end