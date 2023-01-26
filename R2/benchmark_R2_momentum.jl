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
using Dates
using SolverBenchmark

flag = 1

if (flag == 1) # run the whole experiment 
  # T = Float64
  T = Float32

  problems =
    (eval(Meta.parse(problem))(type = Val(T)) for problem ∈ OptimizationProblems.meta[!, :name])

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


  @eval getname(::Type{<:$T}) = $T

  # Write stats into file for future readability
  filename = string(getname(T)) * "_" * Dates.format(now(), "yyyymmdd") * "_stats" # using date in the file name
  save_stats(stats, filename; force=true)

  #write to file 
  for solver ∈ keys(solvers)
    st = string(getname(T)) * string(solver)
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



else # only load the experiment 
  file_to_open = "" #TODO put the file name here
  stats = load_stats(file_to_open)
end



for solver ∈ keys(solvers)
  pretty_stats(stats[solver][!, cols], hdr_override = header)
end



first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)
costnames = ["time", "obj + grad + hess", "iter"]
# costnames = ["time"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_obj .+ df.neval_grad .+ df.neval_hess,
  df -> .!solved(df) .* Inf .+ df.neval_obj,
]
using Plots
gr()
# p = profile_solvers(stats, costs, costnames)
### or using Plots
# pyplot()
p = performance_profile(stats, df -> .!solved(df) .* Inf .+ df.elapsed_time)

for i in 1:3
  if p.series_list[i][:label] == "R2_Momentum_3"
    p.series_list[i][:label] ="β=0.3"
  elseif p.series_list[i][:label] == "R2_Momentum_9"
    p.series_list[i][:label] ="β=0.9"
  elseif p.series_list[i][:label] == "R2"
    p.series_list[i][:label] ="β=0.0"
  end 

end
display(p)

#TODO Then, for the labels, you can check the xlabel and ylabel function applied to pbefore Plots.svg(p, sp)

sp = string(getname(T)) * "_profile"
Plots.svg(p, sp)



