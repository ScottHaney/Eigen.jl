module Eigen

using LinearAlgebra

abstract type IterativeStoppingCritera
end

mutable struct ExactlyNIterations <: IterativeStoppingCritera
    n::Integer
end

mutable struct Residual <: IterativeStoppingCritera
    value
end

function ShouldStop(N::ExactlyNIterations, IterationsExecuted::Integer)
    return IterationsExecuted >= N.n
end

function ShouldStop(R::Residual, IterationsExecuted::Integer)
    return true
end

function rayleighquotient(Matrix::AbstractMatrix, X::AbstractVector)
    LinearAlgebra.transpose(X) * Matrix * x
end

function powermethod(Matrix::AbstractMatrix, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCritera)
    current = Guess
    iteration = 0

    while !ShouldStop(StoppingCriteria, iteration)
        current = LinearAlgebra.normalize!(Matrix * current)
        iteration += 1
    end

    return current
end

end