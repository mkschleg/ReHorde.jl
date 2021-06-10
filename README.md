# ReHorde

Ask me for the data, it is private but can be shared w/ UofA ppl.


To run do the usual:
```julia
julia>]activate .
julia>]instantiate
julia> using ReHorde
julia> ReHorde.main_experiment()
```

To run with threads, start julia with `julia --project -t x` where x is the number of threads.


There is currently no error calculation, and we are just looking at the runtime of the learning portion. The goal is to see how well we do using /mostly/ standard julia. I use Tullio to avoid some allocations to avoid the GC as much as possible. We also ensure we are using single precision floating point numbers everywhere.


The duty cycle (i.e. a single learning step) on my laptop with 8 threads (2 GHz Quad-Core Intel Core i5) is ~7ms.

