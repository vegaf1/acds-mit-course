#Helper Functions 

"""
#pick out the vector part of a quaternion
"""
H = [zeros(1,3); I];

T = [1  zeros(1,3);
     zeros(3,1) -I];

"""
quaternion left multiply
"""
function L(q)
    return [q[1]          -q[2:4]';
            q[2:4]    q[1]*I + hat(q[2:4])]
end

"""    
quaternion right multiply
"""
function R(q)
    return [q[1]          -q[2:4]';
            q[2:4]    q[1]*I - hat(q[2:4])]
end

"""
attitude jacobian
"""
function G(q)
    return L(q)*H
end

"""
quaterion to rotation matrix
"""
function Q(q)
    return H'*L(q)*R(q)'*H
end

"""
hat operator function 
"""
function hat(x)

    x_hat = [0 -x[3] x[2]; x[3] 0 -x[1]; -x[2] x[1] 0]

    return x_hat

end

"""
unhat operator function 
"""
function unhat(x)
    return 0.5*[x[3,2]-x[2,3];
                x[1,3]-x[3,1];
                x[2,1]-x[1,2]]
end

"""
quaternion exponential map 
transform axis-angle to quaterion 
"""

function expq(ϕ)
    θ = norm(ϕ)
    return [cos(θ); ϕ*sinc(θ/π)];
end