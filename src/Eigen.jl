module Eigen

abstract type IterativeStoppingCritera
end

mutable struct ExactlyNIterations <: IterativeStoppingCritera
    n::Integer
end

function ShouldStop(N::ExactlyNIterations, IterationsExecuted::Integer)
    return IterationsExecuted >= N.n
end

function powermethod(Matrix::AbstractMatrix, Guess::AbstractMatrix, StoppingCriteria::IterativeStoppingCritera)
    current = Guess
    iteration = 0

    while !ShouldStop(StoppingCriteria, iteration)
        current = Matrix * current
        iteration = iteration + 1
    end

    return current
end

end