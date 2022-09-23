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
T = Float64
f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])

X = [nlp.meta.x0[1]]
Y = [nlp.meta.x0[2]]
# function cb(nlp, solver)
#   x = solver.x
#   push!(X, x[1])
#   push!(Y, x[2])
#   if solver.output.iter == 20
#     solver.output.status = :user
#   end
# end
println("with β=0.8")
R2(nlp; β = 0.8, verbose = 1)
println("++++++++++++++++++++++++++++++++++\nwith β=0.0")

R2(nlp; verbose = 1)
