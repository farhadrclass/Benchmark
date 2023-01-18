using JSOSolvers # once the PR  #138 is puch remove the comment 
using LinearAlgebra
using Random
using Printf
using DataFrames
using CSV
using OptimizationProblems
using NLPModels
using ADNLPModels
using OptimizationProblems.ADNLPProblems
using SolverCore
using SolverBenchmark
# using GenericExecutionStats
# include("R2.jl")

# T = Float64
T = Float32

problems =
  (eval(Meta.parse(problem))(type = Val(T)) for problem ∈ OptimizationProblems.meta[!, :name])
# (eval(Meta.parse(problem))(type = Val(T)) for problem ∈ OptimizationProblems.meta[!, :name]) # DataFrames and we can choose sub problems

# solvers = Dict(
#   :lbfgs => model -> lbfgs(model, mem=5, atol=1e-5, rtol=0.0),
#   :trunk => model -> trunk(model, atol=1e-5, rtol=0.0),
# )

solvers = Dict(
  :R2_Momentum_3 => nlp -> R2(
    nlp;
    max_time = 60.0,
    # max_eval = 2,
    # atol=T(0.0001),
    β = T(0.3), #testing the momentum 0.9
  ),
  :R2 => nlp -> R2(
    nlp;#atol=T(0.0001),
    max_time = 60.0,
    # max_eval = 2,
  ),
  :R2_Momentum_9 => nlp -> R2(
    nlp;
    # atol=T(0.0001),
    # max_eval = 2,
    max_time = 60.0,
    β = T(0.9), #testing the momentum 0.9
  ),
)

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

@eval getname(::Type{<:$T}) = $T

#write to file 
for solver ∈ keys(solvers)
  st =  string(getname(T)) * string(solver)
  open(st * ".txt", "w") do io
    pretty_stats(io, stats[solver][!, cols], hdr_override = header)
  end
  open(st * ".tex", "w") do io
    println(io, "\\documentclass[varwidth=20cm,crop=true]{standalone}")
    println(io, "\\usepackage{longtable}[=v4.13]")
    println(io, "\\begin{document}")
    pretty_latex_stats(io, stats[solver][!, cols], hdr_override = header)
    println(io, "\\end{document}")
  end
end

first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time", "obj + grad + hess", "iter"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
  df -> .!solved(df) .* Inf .+ df.neval_obj,
]

using Plots
gr()

p = profile_solvers(stats, costs, costnames)
# sp ="//R2//Result//" * string(getname(T)) * string(solver)
sp = string(getname(T)) *"_profile"
Plots.svg(p, sp)