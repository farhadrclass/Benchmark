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

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
nlp = ADNLPModel(f, [-1.2; 1.0])

X = [nlp.meta.x0[1]]
Y = [nlp.meta.x0[2]]
  function cb(nlp, solver)
    x = solver.x
    push!(X, x[1])
    push!(Y, x[2])
    if solver.output.iter == 20
      solver.output.status = :user
    end
  end

R2(nlp,callback = cb; verbose=1)

  