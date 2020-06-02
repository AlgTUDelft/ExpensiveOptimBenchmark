# A port of IDONE to Julia
# Mostly to get an idea of the performance overhead
# of using Python instead.

using LinearAlgebra
using Distributions

struct IDONEModel
    # Dimensionality
    d :: Int64
    # Bounds
    lb :: Vector{Float64}
    ub :: Vector{Float64}
    # Regularization parameter
    reg :: Float64
    # Weights of ReLU basis functions
    # As IDONE is for integers, these weights as well as biases
    # are integers as well. Store as float anyways.
    W :: Array{Float64, 2}
    # Biases of ReLU basis functions
    b :: Array{Float64, 1}
    # Basis function weights. 
    # Updated using Recursive Least Squares
    c :: Vector{Float64}
    Δc :: Vector{Float64}
    # Keep track of the last x update.
    last_x_upd :: Vector{Float64}
    # Recursive Least Squares covariance matrix
    P :: Array{Float64, 2}
end

function generateWb(d :: Int64, lb :: Vector{Float64}, ub :: Vector{Float64})
    @assert d == length(lb)
    @assert d == length(ub)
    
    # Model offset
    n_basis = 1 
    # One variable dependent (incl correction for d = 1)
    n_basis += (2 * round(Int64, sum(ub .- lb))) 
    # Subsequent variable dependent
    n_basis += (2 * round(Int64, sum(1 .+ (ub[1:end-1] .- lb[2:end]) .- (lb[2:end] .- ub[1:end-1]))))

    # Actually construct the matrix
    W = zeros(Float64, (n_basis, d))
    b = zeros(Float64, n_basis)

    # Model offset
    fni = 1
    b[fni] = 1
    fni += 1

    # One variable dependent
    for k in 1:d
        for i in lb[k]:ub[k]
            if i == lb[k]
                W[fni, k] = 1
                b[fni] = -i
                fni += 1
            elseif i == ub[k]
                W[fni, k] = -1
                b[fni] = i
                fni += 1
            else
                W[fni, k] = -1
                b[fni] = i
                W[fni+1, k] = 1
                b[fni+1] = -i
                fni += 2
            end
        end
    end

    # Two subsequent variables
    for k in 2:d
        for i in (lb[k] - ub[k - 1]):(ub[k] - lb[k - 1] + 1)
            if i == lb[k] - ub[k-1]
                W[fni, k] = 1
                W[fni, k - 1] = -1
                b[fni] = -i
                fni += 1
            elseif i == ub[k] - lb[k - 1] + 1
                W[fni, k] = -1
                W[fni, k - 1] = 1
                b[fni] = i
                fni += 1
            else
                W[fni, k] = -1
                W[fni, k - 1] = 1
                b[fni] = i
                W[fni+1, k] = 1
                W[fni+1, k - 1] = -1
                b[fni+1] = -i
                fni += 2
            end
        end
    end

    return W, b
end

function generate_model(d :: Int64, lb :: Vector{Float64}, ub :: Vector{Float64})
    W, b = generateWb(d, lb, ub)
    D = length(b)
    c = ones(D)
    Δc = zeros(D)
    # No initial model offset
    c[1] = 0
    # reg = 1e-3
    reg = 1e-8
    P = Matrix(I / reg, D, D) 
    return IDONEModel(d, lb, ub, reg, W, b, c, Δc, zeros(d), P)
end

function update_model!(model :: IDONEModel, x :: Vector{Float64}, y :: Float64)
    Zx = Z(model, x)
    model.last_x_upd .= x
    PZx = model.P * Zx
    g = PZx ./ (1 + Zx' * PZx)
    model.P .= model.P .- g * PZx'
    model.Δc .= (y - Zx' * model.c) .* g
    # model.c .= model.c .+ (y .- Zx' * model.c) .* g
    model.c .+= model.Δc
end


# Model evaluation 
function id_relu(x)
    return max(0, x)
end

function ∇id_relu(x)
    return (x > 0) + 0.5 * (x == 0)
end

function Z(model :: IDONEModel, x :: Vector{Float64})
    return id_relu.(model.W * x .+ model.b)
end

function ∇Z(model :: IDONEModel, x :: Vector{Float64})
    return ∇id_relu.(model.W * x .+ model.b)
end

function ∇Z!(g :: Vector{Float64}, model :: IDONEModel, x :: Vector{Int64})
    resize!(g, length(model.c))
    mul!(g, model.W, x)
    g .+= model.b
    g .= ∇id_relu.(g)
end

function predict(model :: IDONEModel, x :: Vector{Float64})
    y_pred = dot(model.c, Z(model, x))
    return y_pred
end

function predict_v_and_diff(model :: IDONEModel, x :: Vector{Float64})
    Zx = Z(model, x)
    y_pred = dot(model.c, Zx)
    y_pred_diff = dot(model.Δc, Zx)
    return y_pred, y_pred_diff
end

function ∇predict(model :: IDONEModel, x :: Vector{Float64})
    return model.c' * Diagonal(∇Z(model, x)) * model.W
end

function ∇predict!(g :: Vector{Float64}, model :: IDONEModel, x :: Vector{Float64})
    gz = ∇Z(model, x)
    gz .*= model.c 
    # gz' * W = W' * gz
    # Because apparently the other way around is not recognized.
    # Maybe a bug in Julia?
    mul!(g, model.W', gz)
end

function predict_dist_m2(model :: IDONEModel, x :: Vector{Float64})
    y_pred, y_pred_diff = predict_v_and_diff(model, x)
    return Normal(y_pred, abs(y_pred_diff) / sqrt(2))
end

function predict_EI_fg!(g :: Union{Vector{Float64}, Nothing}, model :: IDONEModel, x :: Vector{Float64}, f_best :: Float64)
    # Mixed Zx, gradient.
    vs = model.W * x .+ model.b
    Zx = id_relu.(vs)
    gZx = ∇id_relu.(vs)

    # Predict y value
    y_pred = dot(model.c, Zx)
    # Predict y variance.
    y_pred_diff = dot(model.Δc, Zx)
    # Predict gradient for y value.
    if !isnothing(g)
        gZx .*= model.c
        g_y_pred = model.W' * gZx
        # And for its' corresponding variance.
        gZx .*= model.Δc
        g_y_pred_diff = model.W' * gZx
    end
    # Estimated distribution
    μ = y_pred

    if x == model.last_x_upd
        #  Special case. Define σ = 0.
        if !isnothing(g)
            g .= g_y_pred
        end
        return y_pred - f_best
    end

    σ = abs(y_pred_diff) / sqrt(2)
    # Update gradient for σ
    if !isnothing(g)
        g_y_pred_diff .*= sign(y_pred_diff)
        g_y_pred_diff .*= sqrt(2)
    end

    # Calculate Expected Improvement and its corresponding gradient.
    n01 = Normal()

    # Calculate expected improvement
    # Calculate difference between computed mu and f_best
    # update gradient g_y_pred to be the gradient of diff_f_best_μ
    diff_f_best_μ = f_best - μ
    
    if !isnothing(g)
        g_y_pred .*= -1
    end
    # 
    diff_f_best_μ_div_sigma = diff_f_best_μ / σ

    # cdf_diff_f_best_μ = cdf(n01, diff_f_best_μ_div_sigma)
    # pdf_diff_f_best_μ = pdf(n01, diff_f_best_μ_div_sigma)
    # 1 - to flip (we want to minimize the value, not maximize.)
    # integral goes from (f_best − μ)/σ to ∞ instead of -∞ to (f_best − μ)/σ. 
    cdf_diff_f_best_μ = 1 - cdf(n01, diff_f_best_μ_div_sigma)
    pdf_diff_f_best_μ = - pdf(n01, diff_f_best_μ_div_sigma)
    ei = diff_f_best_μ * cdf_diff_f_best_μ + σ * pdf_diff_f_best_μ
    
    if !isnothing(g)
        g .= 
            # Grad of diff_f_best_μ * cdf_diff_f_best_μ
            g_y_pred .* cdf_diff_f_best_μ .+
            (diff_f_best_μ * pdf_diff_f_best_μ) .* g_y_pred ./ g_y_pred_diff .+
            # Grad of σ * pdf_diff_f_best_μ
            g_y_pred_diff .* pdf_diff_f_best_μ .+
            (σ * -diff_f_best_μ_div_sigma * pdf_diff_f_best_μ) .* g_y_pred ./ g_y_pred_diff 
    end
    return ei#, g
end

##

function max_EI(μ, σ, f_best)
    if σ < 1e-8
        # Small sigma -> full certainty.
        # Which means μ will be sampled with 100% certainty:
        # eg. it is either a full on improvement (if μ - f_best > 0)
        # or no improvement at all.
        return max(μ - f_best, 0.)
    end
    u = Normal()
    f_best_min_μ = f_best - μ
    f_best_min_μ_div_σ = f_best_min_μ / σ
    return f_best_min_μ * cdf(u, f_best_min_μ_div_σ) + σ * pdf(u, f_best_min_μ_div_σ)
end

function max_EI(n :: Normal, f_best)
    return max_EI(n.μ, n.σ, f_best)
end


function min_EI(μ, σ, f_best)
    if σ < 1e-8
        # Small sigma -> full certainty.
        # Which means μ will be sampled with 100% certainty:
        # eg. it is either a full on improvement (if μ - f_best > 0)
        # or no improvement at all.
        return min(μ - f_best, 0.)
    end
    u = Normal()
    f_best_min_μ = f_best - μ
    f_best_min_μ_div_σ = f_best_min_μ / σ
    return f_best_min_μ * (1. - cdf(u, f_best_min_μ_div_σ)) - σ * pdf(u, f_best_min_μ_div_σ)
end

function min_EI(n :: Normal, f_best)
    return min_EI(n.μ, n.σ, f_best)
end

##
using NLSolversBase
using LineSearches
using Optim

struct IDONE
    model :: IDONEModel
    f :: Function
    best_x :: Vector{Float64}
    best_y :: Ref{Float64}
    y0 :: Float64
    max_iter :: Int64
    current_iter :: Ref{Int64}
end

function scale_f(y0 :: Float64, y :: Float64)
    if abs(y0) > 1e-8
        return (y - y0) / abs(y0)
    else
        return (y - y0)
    end 
end

function inv_scale_f(y0 :: Float64, y :: Float64)
    if abs(y0) > 1e-8
        return y * abs(y0) + y0
    else
        return y + y0
    end 
end

function optimize_surrogate(model :: IDONEModel, initial_x :: Vector{Float64}, f_best :: Float64)
    
    df = OnceDifferentiable(
        (x) -> predict(model, x),
        (g, x) -> ∇predict!(g, model, x),
        ones(model.d)
        )

    # function fg!(F, G, x)
    #     return predict_EI_fg!(G, model, x, f_best)
    # end

    # df = OnceDifferentiable(only_fg!(fg!), ones(model.d))

    # -- Optimize using Optim.jl
    # This seems to occasonally run into errors, likely related
    # to their implementation, as manual inspection indicates the
    # model is well behaved at the point of error.
    # Backtracking Linesearch has issues with NaN values as well.
    # Which seems to be caused by being used together with Fminbox.
    # optimizer = Fminbox()
    optimizer = Fminbox(ParticleSwarm())
    # optimizer = Fminbox(LBFGS(linesearch=LineSearches.BackTracking()))
    options = Optim.Options(iterations=50)
    
    surrogate_solution = nothing
    try
        # surrogate_solution = optimize(df, model.lb, model.ub, initial_x, optimizer, options)
        surrogate_solution = optimize(df, model.lb .- 0.2, model.ub .+ 0.2, initial_x, optimizer, options)
    catch e
        # println("Optimizer failed: $e")
    end
    retries = 0
    while (isnothing(surrogate_solution) || any(isnan.(surrogate_solution.minimizer)) || isnan(surrogate_solution.minimum)) && retries < 10
        try
            surrogate_solution = optimize(df, model.lb, model.ub, initial_x, optimizer, options)
        catch e
            # println("Optimizer failed: $e")
        end
        retries += 1
    end

    if retries >= 10
        return
    end
    # println("Iteration [$(idone.current_iter[])/$(idone.max_iter)] Model indicates $(inv_scale(idone.y0, surrogate_solution.minimum)) at $(surrogate_solution.minimizer).")
    next_x = round.(surrogate_solution.minimizer)
    pred_y = surrogate_solution.minimum

    return next_x, pred_y
end

function variation!(next_x :: Vector{Float64}, idone :: IDONE)
    # Explore if not the last evaluation.
    if idone.current_iter != idone.max_iter
        # Random fuzzy exploration.
        for j in 1:idone.model.d
            # 1/d probability of exploring.
            rand() > 1/idone.model.d && continue
            lbounded = next_x[j] == idone.model.lb[j]
            ubounded = next_x[j] == idone.model.lb[j]
            # Bounds only allow one value.
            lbounded && ubounded && continue
            # Exploration direction is fixed.
            lbounded && (next_x[j] += 1; continue)
            ubounded && (next_x[j] -= 1; continue)
            # Pick randomly otherwise
            next_x[j] += rand(Bool) * 2 - 1
        end
    end
end

function step!(idone :: IDONE)

    # Obtain next point according to model.
    # next_x, pred_y = optimize_surrogate(idone.model, idone.best_x)
    result = optimize_surrogate(idone.model, idone.best_x, idone.best_y[])
    if isnothing(result)
        return
    end

    next_x, pred_y = result

    # Clamp it to be certain.
    next_x .= clamp.(next_x, idone.model.lb, idone.model.ub)
    # Apply variation
    variation!(next_x, idone)

    # @assert all(idone.model.lb .<= next_x)
    # @assert all(next_x .<= idone.model.ub)
    next_x .= clamp.(next_x, idone.model.lb, idone.model.ub)


    # Evaluate function at suggested point next_x
    next_y = idone.f(next_x)
    scaled_next_y = scale_f(idone.y0[], next_y)

    # println("Iteration [$(idone.current_iter[])/$(idone.max_iter)] Scaled $(scaled_next_y) found $(pred_y) predicted")

    # println("Predicted before update: $(predict(idone.model, next_x))")
    update_model!(idone.model, next_x, scaled_next_y)
    # println("Predicted after update: $(predict(idone.model, next_x)), should be $(scaled_next_y)")

    idone.current_iter[] % 1 == 0 &&
        println("Iteration [$(idone.current_iter[])/$(idone.max_iter)] Found value $(next_y) at $(next_x)")

    # Update best
    if idone.best_y[] > next_y
        idone.best_x .= next_x
        idone.best_y[] = next_y
    end

    # Next iteration
    idone.current_iter[] += 1
end

function optimize_IDONE(f :: Function, d :: Int64, lb :: Vector{Float64}, ub :: Vector{Float64}, n_eval=20)
    model = generate_model(d, lb, ub)
    # Start in the middle
    best_x = round.(lb .+ (ub .- lb) ./ 2)
    # println("$(best_x)")
    y0 = f(best_x)
    best_y = y0
    # Note, scaled_y = (y - y0) / abs(y0)
    update_model!(model, best_x, 0.0)

    #
    id = IDONE(model, f, best_x, Ref(best_y), y0, n_eval-1, Ref(1))

    for _ in 1:n_eval-1
        step!(id)
    end

    return id.best_y[], id.best_x, id.model
end

rosen_wrong(x) = sum(100.0 .* (x[1:end] .- x[end:-1:1] .^ 2.0) .^ 2.0 .+ (1 .- x[end:-1:1]) .^ 2.0)
rosen(x) = sum(100.0 .* (x[2:end] .- x[1:end-1] .^ 2.0) .^ 2.0 .+ (1 .- x[1:end-1]) .^ 2.0)

function nqueens(x)
    n = length(x)
    if !(all(x .> 0) && all(x .<= n))
        error("Incorrect indices in $x")
    end
    queens_in_column = zeros(length(x))
    queens_in_primary_diagonal = zeros(2 * length(x) - 1)
    queens_in_perp_diagonal = zeros(2 * length(x) - 1)
    score = 0.
    for (i, p) in enumerate(round.(Int64, x))
        score += queens_in_column[p]
        queens_in_column[p] += 1
        Pdiag = p - i + n
        score += queens_in_primary_diagonal[Pdiag]
        queens_in_primary_diagonal[Pdiag] += 1
        Pdiag = i + p - 1
        score += queens_in_perp_diagonal[Pdiag]
        queens_in_perp_diagonal[Pdiag] += 1
    end
    return score
end