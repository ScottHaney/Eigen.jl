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

mutable struct EigenEstimates
    eigenvalue
    eigenvector
end

function ShouldStop(N::ExactlyNIterations, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    return IterationsExecuted >= N.n
end

function ShouldStop(R::Residual, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    estimate = rayleighquotient(Matrix, V)
    diff = Matrix * V - estimate * V
    return LinearAlgebra.norm(diff) <= R.value
end

function ShouldStop(Composite::CompositeStoppingCriteria, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    for c in Composite.criteria
        if ShouldStop(c, IterationsExecuted, Matrix, V)
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
    iterationmethod2(Matrix, LinearAlgebra.normalize!(Guess), (m,c,i) -> LinearAlgebra.normalize!(m * c), StoppingCriteria)
end

function powermethod(Matrix::AbstractMatrix, Guess::AbstractVector, MaxIterations::Integer, TargetResidual)
    stoppingcriteria = CompositeStoppingCriteria([ExactlyNIterations(MaxIterations), Residual(TargetResidual)])
    powermethod(Matrix, Guess, stoppingcriteria)
end

function inverseiteration(Matrix::AbstractMatrix, Shift, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCritera)
    iterationmethod2(Matrix - LinearAlgebra.UniformScaling(Shift), LinearAlgebra.normalize(Guess), (m, c, i) -> LinearAlgebra.normalize!(m \ c), StoppingCriteria)
end

function inverseiteration(Matrix::AbstractMatrix, Shift, Guess::AbstractVector, MaxIterations::Integer, TargetResidual)
    stoppingcriteria = CompositeStoppingCriteria([ExactlyNIterations(MaxIterations), Residual(TargetResidual)])
    inverseiteration(Matrix, Shift, Guess, stoppingcriteria)
end

end