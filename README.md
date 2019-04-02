Eigen.jl
=========

Julia library for eigenvalue and eigenvector algorithms

## Available Algorithms

### Iterative Algorithms for a Single Eigenvalue/Eigenvector Pair

Each algorithm takes a matrix, some sort of initial value that is specific to the algorithm, and then a structure containing the information about when the algorithm should terminate. The working algorithms are as follows:

* **powermethod** - Power method. Requires an initial vector to run.
  
* **inverseiteration** - Inverse iteration method. Requires an initial vector and a shift value to run.

* **rayleighiteration** - Rayleigh quotient iteration method. Requires an initial vector to run.

To create stopping criteria an instance of the type `IterativeStoppingCriteria` is needed. Currently the stopping criteria can be specified as a residual value (in terms of the 2-norm) and/or a maximum number of iterations. The helper method `stoppingcriteria` can be used to easily create stopping conditions as follows:

```julia
#Stop the algorithm only when the estimated eigenvalue/eigenvector pair is
#within a residual of less than 0.1 according to the 2-norm
stoppingconditions = stoppingcriteria(0.1)

#Stop the algorithm when either the estimated eigenvalue/eigenvector pair is
#within a residusl of less than 0.1 accordind to the 2-norm
#or when 30 iterations have been run
stoppingconditions = stoppingcriteria(0.1, 30)

#Stop the algorithm when 30 iterations have been run
stoppingconditions = ExactlyNIterations(30)
```

Examples for each of the iterative algorithms are given below:

```julia
#Create stopping conditions
stoppingconditions = stoppingcriteria(0.1, 30)

#Assume that we are given a matrix M and a starting initial vector V
M::AbstractMatrix
V::AbstractVector

#Run the power method
result = powermethod(M, V, stoppingconditions)

#Run the inverse iteration method
shift = 5
result = inverseiteration(M, shift, V, stoppingconditions)

#Run the rayleigh quotient iteration method
result = rayleighquotient(M, V, stoppingconditions)

#For each method the return type is the same

#The eigenvalue and eigenvector estimates can be found as follows

result.eigenvalue #eigenvalue
result.eigenvector #eigenvector

#To tell if the iterative algorithm terminated due to finding a result that is within a desired residual
#just check isverified. If it is false that means that the iterative algorithm terminated due to it reaching
#a maximum iteration limit and the result is not known to be within any predefined residual. If it is true
#then the algorithm terminated because it found a result that is within the requested residual

result.isverified
```