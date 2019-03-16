module Eigen

using LinearAlgebra

abstract type IterativeStoppingCritera
end

mutable struct ExactlyNIterations <: IterativeStoppingCritera
    n::Integer
end

function ShouldStop(N::ExactlyNIterations, IterationsExecuted::Integer)
    return IterationsExecuted >= N.n
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