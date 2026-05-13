using Pkg
Pkg.activate(".")

using LinearAlgebra
using Plots
#using MeshCat, GeometryBasics, CoordinateTransformations, Rotations
using Convex, MosekTools
using ForwardDiff

#quaternion functions
function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end

function unhat(S)
    return 0.5*[S[3,2]-S[2,3];
                S[1,3]-S[3,1];
                S[2,1]-S[1,2]]
end

H = [zeros(1,3); I];
T = [1  zeros(1,3);
     zeros(3,1) -I];

function L(q)
    return [q[1]          -q[2:4]';
            q[2:4]    q[1]*I + hat(q[2:4])]
end

function R(q)
    return [q[1]          -q[2:4]';
            q[2:4]    q[1]*I - hat(q[2:4])]
end

function G(q)
    return L(q)*H
end

function Q(q)
    return H'*(R(q)'*L(q))*H
end

function expq(ϕ)
    θ = norm(ϕ)
    return [cos(θ); ϕ*sinc(θ/π)];
end

function logq(q)
    q  = q / norm(q)
    s  = clamp(q[1], -1.0, 1.0)
    θ  = acos(s)
    nv = norm(q[2:4])
    nv < 1e-10 ? zeros(3) : (θ / nv) * q[2:4]
end

#model parameters
J = Diagonal([1.0; 1.25; 1.5])
g = [1 -1 0  0;
     0  0 1 -1;
     0  0 0  0]
w = 0.1;


function Bw(x)
    a = reshape(x[8:19],3,4)
    B = -w*[hat(g[:,1])*a[:,1] hat(g[:,2])*a[:,2] hat(g[:,3])*a[:,3] hat(g[:,4])*a[:,4]]
end


#dynamics
function dynamics(x,u)
    q = x[1:4]/norm(x[1:4])
    ω = x[5:7]
    a = reshape(x[8:19],3,4)
    for k = 1:4
        a[:,k] .= a[:,k]/norm(a[:,k])
    end

    ρ = w*sum(a, dims=2)
    τ = Bw(x)*u
    
    q̇ = 0.5*G(q)*ω
    ω̇ = -J\(hat(ω)*(J*ω + ρ) - τ)
    ȧ = [hat(g[:,1])*a[:,1]*u[1];
         hat(g[:,2])*a[:,2]*u[2];
         hat(g[:,3])*a[:,3]*u[3];
         hat(g[:,4])*a[:,4]*u[4]]

    ẋ = [q̇; ω̇; ȧ]
end


function controller(x)
    kp = 0.5;
    kd = 0.5;

    q = x[1:4]
    #get axis angle error state
    ϕ = logq(q)

    ω = x[5:7]
    
    #pd controller
    τ = -kp*ϕ - kd*ω
end
     

function steering(x,τ)
    B = Bw(x)
    #regularized steering law
    u = B'*((B*B'+1e-3*I)\τ)
end


#rk4
function rkstep(x,u)
    f1 = dynamics(x,u)
    f2 = dynamics(x + 0.5*h*f1,u)
    f3 = dynamics(x + 0.5*h*f2,u)
    f4 = dynamics(x + h*f3,u)
    xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    
    xn[1:4] .= xn[1:4]/norm(xn[1:4]) #re-normalize quaternion

    #re-normalize wheel axes
    xn[8:10] .= xn[8:10]/norm(xn[8:10])
    xn[11:13] .= xn[11:13]/norm(xn[11:13])
    xn[14:16] .= xn[14:16]/norm(xn[14:16])
    xn[17:19] .= xn[17:19]/norm(xn[17:19])
    
    return xn
end



h = 0.1 #time step
n = 1000 #number of time steps
tf = n*h #final time

#sample random attitude
q0 = expq(0.2*randn(3))

ω0 = 0.01*randn(3)

#random initial gimbal angles
a0 = zeros(3,4)
for k = 1:4
    a0[:,k] .= randn(3)
    a0[:,k] .= hat(g[:,k])*a0[:,k]
    a0[:,k] .= a0[:,k]/norm(a0[:,k])
end

#generate random initial conditions
#x0 = [q0; ω0; a0[:]];

#initial condition that leads to a singularity that pseudo inverse can't get out of...
#example from the project
# x0=[0.9042701822532846;
#   0.39813596313876826;
#   0.06317142271220837;
#  -0.14068604655648087;
#   0.005448462332399245;
#   0.0038118416965444903;
#  -0.011059114154415316;
#   0.0;
#  -0.9691739653926361;
#   0.24637740319500392;
#   0.0;
#  -0.20432132079551232;
#   0.9789038757040333;
#   0.5979512060040267;
#   0.0;
#  -0.801532504168315;
#   0.5639729249640095;
#   0.0;
#   0.8257932791610377]


#alternate initial condition
x0 = [0.847226561031405;
  0.0732446800051401;
  0.5258364795266754;
  0.018394780043152908;
 -0.002807278981679324;
  0.020918484382436424;
 -0.00798171207723605;
  0.0;
  0.13174195145061054;
  0.9912840451797784;
  0.0;
  0.9661136333509235;
  0.2581171196443148;
  0.1061036485387192;
  0.0;
 -0.9943550752959286;
  0.8187169663533636;
  0.0;
 -0.5741972910116743];

#Simulate n time steps
xhist = zeros(19,n)
xhist[:,1] .= x0

Bcond = zeros(n)

Bcond[1] = cond(Bw(x0))

#u = zeros(4) #no gimble rates
for k = 1:(n-1)
    u = steering(xhist[:,k], controller(xhist[:,k]))
    xhist[:,k+1] = rkstep(xhist[:,k],u)
    Bcond[k+1] = cond(Bw(xhist[:,k+1]))
end

# -----------------------------------------------------------------------------
#  Control Barrier Function for CMG Singularity Avoidance
# -----------------------------------------------------------------------------


"""cbf singularity measure"""
function singularity_measure(x)
    B = Bw(x)
    δ = 1e-6
    M = B * B' + δ * I(3)
    sqrt(det(M)) - sqrt(δ^3)
end

"""CBF value h(x) = m(x) − ε.  Safe set: h ≥ 0."""
cbf_value(x, ε) = singularity_measure(x) - ε

"""
    lie_derivative_h(x) → c ∈ ℝ⁴

Chain rule:
    ḣ = dh/dt = (∂h/∂x) · ẋ = Σ_k (∂h/∂a[:,k]) · ȧ[:,k]
      = Σ_k (∂h/∂a[:,k]) · (g_k × a_k) · u_k  =  cᵀu

∂h/∂x is computed exactly via ForwardDiff 
h depends on x only through the spin axes x[8:19]; the quaternion and
angular-velocity components have zero gradient.
"""
function lie_derivative_h(x)
    dh_dx = ForwardDiff.gradient(singularity_measure, x)
    a = reshape(x[8:19], 3, 4)
    c = zeros(4)
    for k in 1:4
        dh_da_k = dh_dx[7 + 3*(k-1) + 1 : 7 + 3*k]   # ∂h/∂a[:,k]
        ȧ_k     = hat(g[:, k]) * a[:, k]               # ȧ[:,k] at u_k = 1
        c[k]    = dot(dh_da_k, ȧ_k)
    end
    c
end

"""
    steering_cbf_qp(x, τ; ε, α, u_max) → u

CBF-QP steering: finds the gimbal rates that best track the desired torque
while enforcing the CBF safety constraint:

    minimize  ‖B(x) u − τ‖² + λ‖u‖²        
    subject to  cᵀu + α h(x) ≥ 0          (CBF forward-invariance)
"""

nullspace(Bw(x0))

function steering_cbf_qp(x, τ; ε=0.05, α=1.0)
    
    B     = Bw(x)
    h_val = cbf_value(x, ε)
    c     = lie_derivative_h(x)

    u_var = Variable(4)

    s = Variable(3)

    constraints = Constraint[dot(c, u_var) + α * h_val >= 0]

    #prob = minimize(sumsquares(B * u_var - τ) + λ * sumsquares(u_var), constraints)

    #prob = minimize(norm(s, 1)) + λ * sumsquares(u_var), constraints)

    push!(constraints, B * u_var - τ + s == 0)

    #prob = minimize(100*norm(s, 1) + sumsquares(u_var), constraints)

    nB  = nullspace(B)[:, 1]
    λ   = 1
    #prob = minimize(1000*norm(s, 1) + λ * sumsquares(dot(nB, u_var)), constraints)


    prob = minimize(1000*sumsquares(s) + λ * sumsquares(dot(nB, u_var)), constraints)



    #add in limits 
    # push!(constraints, u_var[1] >= -2)
    # push!(constraints, u_var[1] <= 2)

    # push!(constraints, u_var[2] >= -2)
    # push!(constraints, u_var[2] <= 2)

    # push!(constraints, u_var[3] >= -2)
    # push!(constraints, u_var[3] <= 2)

    # push!(constraints, u_var[4] >= -2)
    # push!(constraints, u_var[4] <= 2)

    #try out 

    solve!(prob, Mosek.Optimizer; silent_solver=true)

    if u_var.value === nothing
        u_fb = steering(x, τ)
        return u_fb, τ - B * u_fb
    end
    return vec(u_var.value), vec(s.value)
end



# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

using Test


@testset "CMG CBF" begin

    # ── Singularity measure ───────────────────────────────────────────────────
    @testset "Singularity measure" begin
        m = singularity_measure(x0)
        @test m isa Float64
        @test m >= 0.0
        println("    m(x0) = $m")

        # Build a well-conditioned state from scratch
        q_r = expq(0.05 * randn(3))
        ω_r = 0.01 * randn(3)
        a_r = zeros(3, 4)
        for k in 1:4
            v = randn(3)
            a_r[:, k] = hat(g[:, k]) * v
            a_r[:, k] ./= norm(a_r[:, k])
        end
        x_r = [q_r; ω_r; a_r[:]]
        @test singularity_measure(x_r) > 0.0
        println("    m(random x) = $(singularity_measure(x_r))")
    end

    # ── CBF value ─────────────────────────────────────────────────────────────
    @testset "CBF value" begin
        ε = 0.05
        h = cbf_value(x0, ε)
        @test h ≈ singularity_measure(x0) - ε atol=1e-12
        println("    h(x0, ε=$ε) = $h")
    end

    # ── CBF gradient consistency check ───────────────────────────────────────
    @testset "Lie derivative (ForwardDiff)" begin
        c = lie_derivative_h(x0)
        @test length(c) == 4
        @test all(isfinite, c)
        # h depends only on spin axes — quaternion/ω gradient must be zero
        dh_dx = ForwardDiff.gradient(singularity_measure, x0)
        @test norm(dh_dx[1:7]) < 1e-12
        println("    c = $c")
    end

    # ── CBF condition satisfied ───────────────────────────────────────────────
    @testset "CBF condition  ḣ + α h ≥ 0" begin
        ε = 0.05;  α = 1.0
        τ_t    = controller(x0)
        u, _   = steering_cbf_qp(x0, τ_t; ε=ε, α=α)
        h   = cbf_value(x0, ε)
        c   = lie_derivative_h(x0)
        ḣ   = dot(c, u)
        @test ḣ + α * h >= -1e-6   # solver tolerance
        println("    ḣ + α h = $(ḣ + α*h)   h = $h")
    end

    # ── QP minimises torque error when constraints are truly inactive ────────
    # Build a well-conditioned state so that the CBF constraint is provably
    # slack at the pseudoinverse: verify cᵀu_nom + α·h > 0 before asserting.
    @testset "QP torque error ≤ u_nom torque error when constraint inactive" begin
        # construct a well-conditioned state far from singularity
        q_r = expq(0.05 * randn(3))
        ω_r = 0.01 * randn(3)
        a_r = zeros(3, 4)
        for k in 1:4
            v = randn(3)
            a_r[:, k] = hat(g[:, k]) * v
            a_r[:, k] ./= norm(a_r[:, k])
        end
        x_r = [q_r; ω_r; a_r[:]]

        ε = 0.01;  α = 1.0
        τ_t   = controller(x_r)
        u_nom = steering(x_r, τ_t)
        h_val = cbf_value(x_r, ε)
        c     = lie_derivative_h(x_r)

        # only run the QP comparison when u_nom already satisfies the CBF
        # (guarantees the constraint is inactive at the optimum)
        if dot(c, u_nom) + α * h_val > 1e-3
            u_qp, _ = steering_cbf_qp(x_r, τ_t; ε=ε, α=α)
            B       = Bw(x_r)
            err_qp  = norm(B * u_qp  - τ_t)
            err_nom = norm(B * u_nom - τ_t)
            @test err_qp <= err_nom + 1e-6
            println("    torque err — QP: $err_qp   u_nom: $err_nom   (constraint slack: $(dot(c,u_nom)+α*h_val))")
        else
            println("    skipped: CBF constraint active for u_nom, cannot compare fairly")
            @test true
        end
    end

    # ── Simulation: CBF-QP keeps cond(B) bounded ─────────────────────────────
    @testset "Simulation safety" begin
        m0 = singularity_measure(x0)
        ε  = m0 * 0.5
        α  = 1.0
        println("    m(x0) = $m0,  using ε = $ε")
        n_sim = 1000
        xno  = zeros(19, n_sim);  xno[:, 1]  .= x0
        xcbf = zeros(19, n_sim);  xcbf[:, 1] .= x0
        cno  = zeros(n_sim);  ccbf = zeros(n_sim)
        cno[1] = ccbf[1] = cond(Bw(x0))

        for k in 1:(n_sim - 1)
            xno[:, k+1]  = rkstep(xno[:, k], steering(xno[:, k], controller(xno[:, k])))
            u_k, _ = steering_cbf_qp(xcbf[:, k], controller(xcbf[:, k]); ε=ε, α=α)
            xcbf[:, k+1] = rkstep(xcbf[:, k], u_k)
            cno[k+1]  = cond(Bw(xno[:, k+1]))
            ccbf[k+1] = cond(Bw(xcbf[:, k+1]))
        end

        println("    Max cond(B) — no CBF: $(round(maximum(cno),  digits=1))" *
                "    CBF-QP: $(round(maximum(ccbf), digits=1))")
        @test maximum(ccbf) < maximum(cno)
        @test maximum(ccbf) < 1e4
    end

end

nb = nullspace(Bw(x0)) 

# -----------------------------------------------------------------------------
#  Simulation + Plots
#
#  run_comparison()  →  simulates CBF-QP and nominal (no CBF) from x0
#  plot_results(r)   →  six-panel figure covering safety and tracking
# -----------------------------------------------------------------------------

function logq_safe(q)
    q  = q / norm(q)
    s  = clamp(q[1], -1.0, 1.0)
    θ  = acos(s)
    nv = norm(q[2:4])
    nv < 1e-10 ? zeros(3) : (θ / nv) * q[2:4]
end

function run_comparison(; ε=nothing, α=2, n_sim=1000)
    m0  = singularity_measure(x0)
    ε   = isnothing(ε) ? m0 * 0.5 : ε
    t   = (0:n_sim-1) .* h   # time vector (h is the global timestep)

    # Pre-allocate histories: rows = quantities, cols = time
    q_no   = zeros(4,  n_sim);  q_no[:,  1] .= x0[1:4]
    q_cbf  = zeros(4,  n_sim);  q_cbf[:, 1] .= x0[1:4]
    ω_no   = zeros(3,  n_sim);  ω_no[:,  1] .= x0[5:7]
    ω_cbf  = zeros(3,  n_sim);  ω_cbf[:, 1] .= x0[5:7]
    u_no   = zeros(4,  n_sim)
    u_cbf  = zeros(4,  n_sim)
    m_no   = zeros(n_sim);  m_no[1]  = m0
    m_cbf  = zeros(n_sim);  m_cbf[1] = m0
    h_no   = zeros(n_sim);  h_no[1]  = m0 - ε
    h_cbf  = zeros(n_sim);  h_cbf[1] = m0 - ε
    τerr_no  = zeros(n_sim)
    τerr_cbf = zeros(n_sim)
    s_cbf    = zeros(3, n_sim)   # QP slack variable (torque error) at each step
    xcbf_hist = zeros(19, n_sim) # full CBF state history for null-space check
    xcbf_hist[:, 1] .= x0

    xno  = copy(x0)
    xcbf = copy(x0)

    for k in 1:(n_sim - 1)
        τ_no  = controller(xno)
        τ_cbf = controller(xcbf)

        un       = steering(xno, τ_no)
        uc, sc   = steering_cbf_qp(xcbf, τ_cbf; ε=ε, α=α)

        τerr_no[k+1]  = norm(Bw(xno)  * un - τ_no)
        τerr_cbf[k+1] = norm(Bw(xcbf) * uc - τ_cbf)
        s_cbf[:, k+1] .= sc

        xno  = rkstep(xno,  un)
        xcbf = rkstep(xcbf, uc)

        q_no[:,  k+1] .= xno[1:4]
        q_cbf[:, k+1] .= xcbf[1:4]
        ω_no[:,  k+1] .= xno[5:7]
        ω_cbf[:, k+1] .= xcbf[5:7]
        u_no[:,  k+1] .= un
        u_cbf[:, k+1] .= uc
        xcbf_hist[:, k+1] .= xcbf
        m_no[k+1]  = singularity_measure(xno)
        m_cbf[k+1] = singularity_measure(xcbf)
        h_no[k+1]  = m_no[k+1]  - ε
        h_cbf[k+1] = m_cbf[k+1] - ε
    end

    # Attitude error: rotation angle from identity  φ = ‖logq(q)‖
    φ_no  = [norm(logq_safe(q_no[:,  k])) for k in 1:n_sim]
    φ_cbf = [norm(logq_safe(q_cbf[:, k])) for k in 1:n_sim]

    # ── Null-space diagnostic ────────────────────────────────────────────────
    δu_null_norm = zeros(n_sim)
    δu_row_norm  = zeros(n_sim)
    for k in 1:n_sim
        Bk  = Bw(xcbf_hist[:, k])
        τk  = controller(xcbf_hist[:, k])
        nBk = nullspace(Bk)[:, 1]
        u_exact = Bk' * ((Bk * Bk') \ τk)        # exact pseudoinverse (no regularisation)
        δu  = u_cbf[:, k] - u_exact
        δu_null_norm[k] = abs(dot(δu, nBk))       # null-space component magnitude
        δu_row_norm[k]  = norm(δu - dot(δu, nBk) * nBk)  # row-space component magnitude
    end

    (t=t, ε=ε,
     m_no=m_no, m_cbf=m_cbf,
     h_no=h_no, h_cbf=h_cbf,
     φ_no=φ_no, φ_cbf=φ_cbf,
     ω_no=ω_no, ω_cbf=ω_cbf,
     u_no=u_no, u_cbf=u_cbf,
     τerr_no=τerr_no, τerr_cbf=τerr_cbf,
     s_cbf=s_cbf, xcbf_hist=xcbf_hist,
     δu_null_norm=δu_null_norm, δu_row_norm=δu_row_norm,
     xno=xno, xcbf=xcbf)
end

function plot_results(r)
    t = r.t

    # ── 1. Manipulability ─────────────────────────────────────────────────────
    p1 = plot(t, r.m_no,  label="no CBF",  color=:firebrick, lw=1.5,
              ylabel="m(x)",  title="Manipulability",  legend=:topright)
    plot!(p1, t, r.m_cbf, label="CBF-QP", color=:steelblue,  lw=1.5)
    hline!(p1, [r.ε], label="ε (safe boundary)", color=:black, ls=:dash)

    # ── 2. CBF value  h = m − ε ───────────────────────────────────────────────
    p2 = plot(t, r.h_no,  label="no CBF",  color=:firebrick, lw=1.5,
              ylabel="h(x) = m − ε", title="CBF value  (h ≥ 0 = safe)",
              legend=:topright)
    plot!(p2, t, r.h_cbf, label="CBF-QP", color=:steelblue,  lw=1.5)
    hline!(p2, [0.0], label="h = 0", color=:black, ls=:dash)

    # ── 3. Attitude error ─────────────────────────────────────────────────────
    p3 = plot(t, r.φ_no,  label="no CBF",  color=:firebrick, lw=1.5,
              ylabel="‖logq(q)‖ (rad)", title="Attitude error",
              legend=:topright)
    plot!(p3, t, r.φ_cbf, label="CBF-QP", color=:steelblue,  lw=1.5)

    # ── 4. Angular velocity norm ──────────────────────────────────────────────
    ωnorm_no  = vec(sqrt.(sum(r.ω_no  .^ 2, dims=1)))
    ωnorm_cbf = vec(sqrt.(sum(r.ω_cbf .^ 2, dims=1)))
    p4 = plot(t, ωnorm_no,  label="no CBF",  color=:firebrick, lw=1.5,
              ylabel="‖ω‖ (rad/s)", title="Angular velocity magnitude",
              legend=:topright)
    plot!(p4, t, ωnorm_cbf, label="CBF-QP", color=:steelblue,  lw=1.5)

    # ── 5. Gimbal rate magnitude ──────────────────────────────────────────────
    unorm_no  = vec(sqrt.(sum(r.u_no  .^ 2, dims=1)))
    unorm_cbf = vec(sqrt.(sum(r.u_cbf .^ 2, dims=1)))
    p5 = plot(t, unorm_no,  label="no CBF",  color=:firebrick, lw=1.5,
              ylabel="‖u‖ (rad/s)", title="Gimbal rate magnitude",
              legend=:topright, yscale=:identity)
    plot!(p5, t, unorm_cbf, label="CBF-QP", color=:steelblue,  lw=1.5)

    # ── 6. Torque tracking error ──────────────────────────────────────────────
    p6 = plot(t, r.τerr_no,  label="no CBF",  color=:firebrick, lw=1.5,
              ylabel="‖Bu − τ_des‖", title="Torque tracking error",
              legend=:topright)
    plot!(p6, t, r.τerr_cbf, label="CBF-QP", color=:steelblue,  lw=1.5)

    fig = plot(p1, p2, p3, p4, p5, p6;
               layout=(3, 2), size=(900, 900),
               xlabel="Time (s)",
               bottom_margin=4Plots.mm, left_margin=6Plots.mm)

    display(fig)
    fig
end

# # Run and plot:
r = run_comparison()

plot_results(r)

#savefig("cmg_results_test3.png")


# plot(r.t, r.δu_null_norm, label="null-space correction")
# plot!(r.t, r.δu_row_norm,  label="row-space correction (= torque error)")
# plot!(r.t, r.τerr_cbf,     label="‖s‖", ls=:dash)


r.τerr_cbf[1:100]  


plot(r.τerr_cbf)

size(r.τerr_cbf)

plot(r.xcbf[1,:])


r.xcbf[1,:]

size(r.xcbf_hist)

plot(r.xcbf_hist[1,:])
plot(r.xcbf_hist[2,:])
plot(r.xcbf_hist[3,:])
plot(r.xcbf_hist[4,:])


nullspace(Bw(x0))

plot(r.u_cbf[:,1:50]')


plot(r.u_no')
