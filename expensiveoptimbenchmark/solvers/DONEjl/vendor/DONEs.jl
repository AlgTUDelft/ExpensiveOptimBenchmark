module DONEs

using Distributions
using LinearAlgebra
using NLopt

include("RFEs.jl")

export RFE, DONE, add_measurement!, update_optimal_input!, new_input, evaluateRFE

"""
    DONE(
        rfe::RFE # Random Feature/Fourier Expansion

        # Variables
        current_optimal_x::Vector{T} where T <: AbstractFloat
        n::Int # number of variables
        lower_bound::Vector{T} where T <: AbstractFloat
        upper_bound::Vector{T} where T <: AbstractFloat

        # DONE algorithm sliding window variant
        sliding_window::Bool
        sliding_window_length::Int
        past_inputs::Vector{Vector{T}} where T <: AbstractFloat
        past_outputs::Vector{T} where T <: AbstractFloat

        # Algorithm variables
        iteration::Int

        # Exploration
        surrogate_exploration_prob_dist::Distributions.Distribution
        function_exploration_prob_dist::Distributions.Distribution)

A DONE (Data-based Online Non-linear Extremun-seeker) struct is the main structure for finding the minimum of an unknown function that is typically expensive or difficult to evaluate and that returns a
'measurement' that is corrupted by noise.
The purpose of the underlying algorithm is to find the minimum of this function in as few measurements as possible.

For details of the underlying algorithm, see:
L. Bliek, H.R.G.W. Verstraete, M. Verhaegen and S. Wahls - Online otimization with costly and noisy measurements using random Fourier expansions. (https://arxiv.org/abs/1603.09620)
L. Bliek - Automatic Tuning of Photonic Beamformers.

The (unknown) function f(x) that we want to minimize is approximated using a RFE (Random Fourier/Feature Expansion, type `?RFE`).
n = length(x) and lower_bound .< x .< upper_bound

If f changes over time, then a sliding window of measurements can be used.

The struct keeps track of the current estimate of the optimal x. This is computed from the estimated function (based on the RFE) and a minimization using the NLopt package and the L-BFGS algorithm, initialized using the currently estimated optimal x + some perturbation from the surrogate_exploration_prob_dist probability distribution.

Finding the minimum of the function is balanced with staying at the minimum of the function through the use of exploration of the function using the function_exploration_prob_dist probability distribution to suggest a new measurement around the optimal x + some perturbation.
"""
mutable struct DONE
    rfe::RFE # Random Feature/Fourier Expansion

    # Variables
    current_optimal_x::Vector{T} where T <: AbstractFloat
    n::Int # number of variables
    lower_bound::Vector{T} where T <: AbstractFloat
    upper_bound::Vector{T} where T <: AbstractFloat

    # DONE algorithm sliding window variant
    sliding_window::Bool
    sliding_window_length::Int
    past_inputs::Vector{Vector{T}} where T <: AbstractFloat
    past_outputs::Vector{T} where T <: AbstractFloat

    # Algorithm variables
    iteration::Int

    # Exploration
    surrogate_exploration_prob_dist::Distributions.Distribution
    function_exploration_prob_dist::Distributions.Distribution
end

"""
    DONE(rfe, lower_bound, upper_bound, σ_surrogate_exploration, σ_function_exploration; sliding_window=false, sliding_window_length=1 )

The surrogate explaration and the function exploration are initialized with a probability distribution N(0,σI).
sliding_window_length is the number of past measurements to keep into account.
"""
function DONE(rfe, lower_bound, upper_bound, σ_surrogate_exploration, σ_function_exploration; sliding_window=false, sliding_window_length=1 )
    n = size(rfe.Ω,2)
    current_optimal_x = (lower_bound+upper_bound)./2.0
    if sliding_window
        past_inputs = Vector{Float64}[]
        past_outputs = Float64[]
    else
        sliding_window_length=0
        past_inputs = Vector{Float64}[]
        past_outputs = Float64[]
    end
    iteration = 0
    surrogate_exploration_prob_dist = Distributions.MvNormal(zeros(Float64,n),σ_surrogate_exploration*Diagonal(ones(Float64,n)))
    function_exploration_prob_dist = Distributions.MvNormal(zeros(Float64,n),σ_function_exploration*Diagonal(ones(Float64,n)))

    DONE(rfe,current_optimal_x,n,lower_bound,upper_bound,sliding_window,
        sliding_window_length,past_inputs,past_outputs,iteration,
        surrogate_exploration_prob_dist,function_exploration_prob_dist)
end

"""
    add_measurement!(alg::DONE,x::Vector{T} where T <: AbstractFloat,y::AbstractFloat)

Process a new measurement y from a function evaluation f(x).

Implementation details: see page 111 of L. Bliek - Automatic Tuning of Photonic Beamformers
"""
function add_measurement!(alg::DONE,x::Vector{T} where T <: AbstractFloat,y::AbstractFloat)
    @assert length(x) == alg.n

    v = alg.rfe.variable_offset ? alg.rfe.offset : 0.

    # downdate with oldest measurement
    if alg.sliding_window && alg.sliding_window_length + 1 <= alg.iteration
        a, g = downdateRFE!(alg.rfe, alg.past_inputs[1], alg.past_outputs[1]+v)
        if alg.rfe.variable_offset
            alg.rfe.h[:] = alg.rfe.h - g*(1.0 - dot(a,alg.rfe.h))
        end

        # update list of inputs and measurements
        alg.past_inputs[:] = vcat(alg.past_inputs[2:end],[x])
        alg.past_outputs[:] = vcat(alg.past_outputs[2:end],y)
    elseif alg.sliding_window && alg.sliding_window_length + 1 > alg.iteration
        alg.past_inputs[:] = push!(alg.past_inputs,x)
        alg.past_outputs[:] = push!(alg.past_outputs,y)
    end

    # account for variable offset if any
    if alg.rfe.variable_offset && y + v > 0
        alg.rfe.offset = -2y
        alg.rfe.c[:] = alg.rfe.c + (-2y - v)*alg.rfe.h
        v = -2y
    end

    # update with newest measurement
    a,g = updateRFE!(alg.rfe,x,y+v)

    # account for variable offset if any
    if alg.rfe.variable_offset
        alg.rfe.h[:] = alg.rfe.h + g*(1.0 - dot(a,alg.rfe.h))
    end

    alg.iteration = alg.iteration + 1
end

function project_on_bounds(x,lb,ub)
    return [min(max(xi,lbi),ubi) for (xi,lbi,ubi) in zip(x,lb,ub)]
end

"""
    x_opt = update_optimal_input!(alg::DONE)

Based on the current estimate of the function to be minimized, compute (estimate of) the minimizing argument.
"""
function update_optimal_input!(alg::DONE)
    Ω = alg.rfe.Ω
    b = alg.rfe.b
    c = alg.rfe.c
    o = alg.rfe.offset

    f(x) = dot(c, cos.(Ω*x + b)) + o
    ∇f(x) = -Ω' * Diagonal(sin.(Ω*x + b)) * c
    function myfunc(x::Vector,grad::Vector)
        grad[:] = ∇f(x)
        return f(x)
    end

    opt = NLopt.Opt(:LD_LBFGS,alg.n)
    opt.lower_bounds = alg.lower_bound
    opt.upper_bounds = alg.upper_bound
    opt.min_objective = myfunc

    x0 = project_on_bounds(alg.current_optimal_x + rand(alg.surrogate_exploration_prob_dist),alg.lower_bound,alg.upper_bound)
    (minf,minx,ret) = NLopt.optimize(opt, x0)
    alg.current_optimal_x[:] = minx
    return minx
end

"""
    x_new = new_input(alg::DONE)

Generate a preffered new measurement to be done on the (noisy) functino f.
"""
function new_input(alg::DONE)
    return project_on_bounds(alg.current_optimal_x + rand(alg.function_exploration_prob_dist),alg.lower_bound,alg.upper_bound)
end

# Added as reference.
function minimize_DONE(f :: Function, N :: Int64, lb :: Vector{Float64}, ub :: Vector{Float64}, σs :: Float64 = 0.1, σf :: Float64 = 0.1)
    n_basis = 3
    rfe = RFE(20, n_basis, 0.1)
    done = DONE(rfe, lb, ub, σs, σf)
    best_x :: Union{Nothing, Vector{Float64}} = nothing
    best_y :: Float64 = typemax(Float64)
    for i in 1:N
        xi = new_input(done)
        yi = f(xi)
        if yi < best_y
            best_x = xi
            best_y = yi
        end
        add_measurement!(done, xi, yi)
        update_optimal_input!(done)
    end
    return best_y, best_x
end

end # module
