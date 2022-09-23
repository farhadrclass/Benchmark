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
# include("R2.jl")

# T = Float64
T=  Float32

problems =
  (eval(Meta.parse(problem))(type = Val(T)) for problem ∈ OptimizationProblems.meta[!, :name])

# solvers = Dict(
#   :lbfgs => model -> lbfgs(model, mem=5, atol=1e-5, rtol=0.0),
#   :trunk => model -> trunk(model, atol=1e-5, rtol=0.0),
# )

solvers = Dict(:R2_Momentum_9 => nlp -> R2(
  nlp;
  max_eval=10,
  β = T(0.9), #testing the momentum 0.9
), :R2 => nlp -> R2(nlp;max_eval=10))

stats = bmark_solvers(
  solvers,
  problems,
  skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5),
)

cols = [
  :id,
  :name,
  :nvar,
  :objective,
  :dual_feas,
  :neval_obj,
  :neval_grad,
  :neval_hess,
  :iter,
  :elapsed_time,
  :status,
]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_obj => "# f",
  :neval_grad => "# ∇f",
  :neval_hess => "# ∇²f",
  :elapsed_time => "t",
)

for solver ∈ keys(solvers)
  pretty_stats(stats[solver][!, cols], hdr_override = header)
end

# meta = OptimizationProblems.meta
# names_pb_vars = meta[(meta.variable_nvar .== true) .& (meta.ncon .== 0), [:nvar, :name]]
# adproblems = (
#   eval(Meta.parse("ADNLPProblems.$(pb[:name])()")) for pb in eachrow(names_pb_vars)
# )
# load the problems:
# problems = (
#   eval(problem)(type = Val(T)) for
#   problem ∈ setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
# )

# println("Getting the problems")
# # Create a dictionary for the solver: 
# solver = Dict(
#         # :R2_Momentum_9 => 
#         #     nlp -> R2(
#         #         nlp;
#         #         β=0.9, #testing the momentum 0.9
#         #         ),
#         :R2 => 
#             nlp -> R2(
#                 nlp;
#                 verbose=1
#             )
#             )

# println("Starting the benchmark")
# # benchmark the problems
# stats = bmark_solvers(solver, problems, skipif = prob -> (!unconstrained(prob) || get_nvar(prob) > 20 || get_nvar(prob) < 15))

# columns = [:id, :name, :nvar, :objective, :dual_feas, :neval_obj, :neval_grad, :neval_hess, :iter, :elapsed_time, :status]
# header = Dict(
#   :nvar => "n",
#   :objective => "f(x)",
#   :dual_feas => "‖∇f(x)‖",
#   :neval_obj => "# f",
#   :neval_grad => "# ∇f",
#   :neval_hprod => "# ∇²f v",
#   :elapsed_time => "t",
# )
# println("Print the stats")
# for solver ∈ keys(solver)
#     pretty_stats(stats[solver][!, columns], hdr_override=header)
# end

# # first_order(df) = df.status .== :first_order
# # unbounded(df) = df.status .== :unbounded
# # solved(df) = first_order(df) .| unbounded(df)
# # costnames = ["time", "obj + grad + hess"]
# # costs = [
# #   df -> .!solved(df) .* Inf .+ df.elapsed_time,
# #   df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
# # ]

# # using Plots
# # gr()

# # profile_solvers(stats, costs, costnames)
