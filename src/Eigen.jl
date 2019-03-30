module Eigen

using LinearAlgebra

abstract type IterativeStoppingCriteria
end

mutable struct ExactlyNIterations <: IterativeStoppingCriteria
    n::Integer
end

mutable struct Residual <: IterativeStoppingCriteria
    value
end

mutable struct CompositeStoppingCriteria <: IterativeStoppingCriteria
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

    EigenEstimates(Matrix::AbstractMatrix, Guess::AbstractVector) = new(rayleighquotient(Matrix, Guess), Guess)
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

function iterationmethod(Matrix::AbstractMatrix, StartingValues, IterationAction::Function, StoppingCriteria::IterativeStoppingCriteria)
    current = StartingValues
    iteration = 0

    while !ShouldStop(StoppingCriteria, iteration, Matrix, current)
        current = IterationAction(Matrix, current, iteration)
        iteration += 1
    end

    return iterationresult(Matrix, current)
end

function iterationresult(Matrix::AbstractMatrix, V::AbstractVector)
    EigenEstimates(Matrix, V)
end

function iterationresult(Matrix::AbstractMatrix, Estimates::EigenEstimates)
    Estimates
end

function powermethod(Matrix::AbstractMatrix, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCriteria)
    iterationmethod(Matrix, LinearAlgebra.normalize!(Guess), (m,c,i) -> LinearAlgebra.normalize!(m * c), StoppingCriteria)
end

function inverseiteration(Matrix::AbstractMatrix, Shift, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCriteria)
    iterationmethod(Matrix - LinearAlgebra.UniformScaling(Shift), LinearAlgebra.normalize(Guess), (m, c, i) -> LinearAlgebra.normalize!(m \ c), StoppingCriteria)
end

function rayleighiteration(Matrix::AbstractMatrix, Guess::AbstractVector, StoppingCriteria::IterativeStoppingCriteria)
    iterationmethod(Matrix, EigenEstimates(Matrix, Guess),
    function(m,c,i)
        newguess = LinearAlgebra.normalize!((m - LinearAlgebra.UniformScaling(c.eigenvalue)) \ c.eigenvector)
        EigenEstimates(m, newguess)
    end, StoppingCriteria)
end

end