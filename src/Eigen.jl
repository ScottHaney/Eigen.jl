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

function ShouldStop(N::ExactlyNIterations, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    return IterationsExecuted >= N.n
end

function ShouldStop(R::Residual, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    estimate = rayleighquotient(Matrix, V)
    diff = Matrix * V - estimate * V
    return LinearAlgebra.norm(diff) <= R.value
end

function rayleighquotient(Matrix::AbstractMatrix, X::AbstractVector)
    LinearAlgebra.transpose(X) * Matrix * X
end

function powermethod(Matrix::AbstractMatrix, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCritera)
    current = LinearAlgebra.normalize(Guess)
    iteration = 0

    while !ShouldStop(StoppingCriteria, iteration, Matrix, current)
        current = LinearAlgebra.normalize!(Matrix * current)
        iteration += 1
    end

    return current
end

end