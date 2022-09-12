using JSOSolvers # once the PR  #138 is puch remove the comment 
using LinearAlgebra
using Random
using Printf
using DataFrames
using OptimizationProblems
using NLPModels
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using SolverCore
using SolverBenchmark
# using GenericExecutionStats
include("R2.jl")


T=Float64

# load the problems:
problems = (
  eval(problem)(type = Val(T)) for
  problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
)


# Create a dictionary for the solver: 
solver = Dict(
        :R2_Momentum_9 => 
            nlp -> R2(
                nlp;
                maxiterations=1000,
                η1 = 0.3,
                η2 = 0.7,
                γ1 = 1 / 2,
                γ2 = 2.0,
                ϵ_abs = 1e-6,
                ϵ_rel = 1e-6,
                β=0.9, #testing the momentum 0.9
                verbose = false),
        :R2 => 
            nlp -> R2(
                nlp;
                maxiterations=1000,
                η1 = 0.3,
                η2 = 0.7,
                γ1 = 1 / 2,
                γ2 = 2.0,
                ϵ_abs = 1e-6,
                ϵ_rel = 1e-6,
                β=0.0, #testing the momentum 0.0
                verbose = false)
            )



# benchmark the problems
stats = bmark_solvers(solver, problems, skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5))

columns = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :iter, :elapsed_time, :status]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_obj => "# f",
  :neval_grad => "# ∇f",
  :neval_hprod => "# ∇²f v",
  :elapsed_time => "t",
)

for solver ∈ keys(solver)
    pretty_stats(stats[solver][!, columns], hdr_override=header)
end


# first_order(df) = df.status .== :first_order
# unbounded(df) = df.status .== :unbounded
# solved(df) = first_order(df) .| unbounded(df)
# costnames = ["time", "obj + grad + hess"]
# costs = [
#   df -> .!solved(df) .* Inf .+ df.elapsed_time,
#   df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
# ]

# using Plots
# gr()

# profile_solvers(stats, costs, costnames)