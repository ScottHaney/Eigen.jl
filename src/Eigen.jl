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

mutable struct CompositeStoppingCriteria <: IterativeStoppingCritera
    criteria
end

function stoppingcriteria(TargetResidual::Real)
    Residual(TargetResidual)
end

function stoppingcriteria(TargetResidual::Real, MaxIterations::Integer)
    CompositeStoppingCriteria([ExactlyNIterations(MaxIterations), Residual(TargetResidual)])
end

mutable struct EigenEstimates
    eigenvalue::Number
    eigenvector::AbstractVector
end

function createeigenestimates(Matrix::AbstractMatrix, Guess::AbstractVector)
    EigenEstimates(rayleighquotient(Matrix, Guess), Guess)
end

function ShouldStop(N::ExactlyNIterations, IterationsExecuted::Integer, Matrix::AbstractMatrix, CurrentValues)
    return IterationsExecuted >= N.n
end

function ShouldStop(R::Residual, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    estimate = rayleighquotient(Matrix, V)
    diff = Matrix * V - estimate * V
    return LinearAlgebra.norm(diff) <= R.value
end

function ShouldStop(R::Residual, IterationsExecuted::Integer, Matrix::AbstractMatrix, CurrentValues::EigenEstimates)
    diff = Matrix * CurrentValues.eigenvector - CurrentValues.eigenvalue * CurrentValues.eigenvector
    return LinearAlgebra.norm(diff) <= R.value
end

function ShouldStop(Composite::CompositeStoppingCriteria, IterationsExecuted::Integer, Matrix::AbstractMatrix, CurrentValues)
    for c in Composite.criteria
        if ShouldStop(c, IterationsExecuted, Matrix, CurrentValues)
            return true
        end
    end

    return false
end

function rayleighquotient(Matrix::AbstractMatrix, X::AbstractVector)
    LinearAlgebra.transpose(X) * Matrix * X
end

function iterationmethod(Matrix::AbstractMatrix, StartingValues, IterationAction::Function, StoppingCriteria::IterativeStoppingCritera)
    current = StartingValues
    iteration = 0

    while !ShouldStop(StoppingCriteria, iteration, Matrix, current)
        current = IterationAction(Matrix, current, iteration)
        iteration += 1
    end

    return current
end

function powermethod(Matrix::AbstractMatrix, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCritera)
    iterationmethod(Matrix, LinearAlgebra.normalize!(Guess), (m,c,i) -> LinearAlgebra.normalize!(m * c), StoppingCriteria)
end

function inverseiteration(Matrix::AbstractMatrix, Shift, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCritera)
    iterationmethod(Matrix - LinearAlgebra.UniformScaling(Shift), LinearAlgebra.normalize(Guess), (m, c, i) -> LinearAlgebra.normalize!(m \ c), StoppingCriteria)
end

function rayleighiteration(Matrix::AbstractMatrix, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCritera)
    iterationmethod(Matrix, createeigenestimates(Matrix, Guess),
    function(m,c,i)
        newguess = LinearAlgebra.normalize!((m - LinearAlgebra.UniformScaling(c.eigenvalue)) \ c.eigenvector)
        createeigenestimates(m, newguess)
    end, StoppingCriteria)
end

end