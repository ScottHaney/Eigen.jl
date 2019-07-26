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

mutable struct EigenEstimates
    eigenvalue::Number
    eigenvector::AbstractVector

    EigenEstimates(Matrix::AbstractMatrix, Guess::AbstractVector) = new(rayleighquotient(Matrix, Guess), Guess)
    EigenEstimates(val::Number, vec::AbstractVector) = new(val, vec)
end

function normalizeStrategy(x::AbstractVector)
    return LinearAlgebra.normalize!(x)
end

function normalizeStrategy(estimates::EigenEstimates)
    estimates.eigenvector = normalizeStrategy(estimates.eigenvector)
    return estimates
end

function stoppingcriteria(TargetResidual::Real)
    Residual(TargetResidual)
end

function stoppingcriteria(TargetResidual::Real, MaxIterations::Integer)
    CompositeStoppingCriteria([ExactlyNIterations(MaxIterations), Residual(TargetResidual)])
end

mutable struct EigenResult
    eigenvalue::Number
    eigenvector::AbstractVector
    isverified::Bool

    EigenResult(Estimates::EigenEstimates, isverified::Bool) = new(Estimates.eigenvalue, Estimates.eigenvector, isverified)
end

function shouldstop(N::ExactlyNIterations, IterationsExecuted::Integer, Matrix::AbstractMatrix, CurrentValues)
    return IterationsExecuted >= N.n, false
end

function shouldstop(R::Residual, IterationsExecuted::Integer, Matrix::AbstractMatrix, V::AbstractVector)
    estimate = rayleighquotient(Matrix, V)
    diff = Matrix * V - estimate * V
    return LinearAlgebra.norm(diff) <= R.value, true
end

function shouldstop(R::Residual, IterationsExecuted::Integer, Matrix::AbstractMatrix, CurrentValues::EigenEstimates)
    diff = Matrix * CurrentValues.eigenvector - CurrentValues.eigenvalue * CurrentValues.eigenvector
    return LinearAlgebra.norm(diff) <= R.value, true
end

function shouldstop(Composite::CompositeStoppingCriteria, IterationsExecuted::Integer, Matrix::AbstractMatrix, CurrentValues)
    for c in Composite.criteria
        stop, foundresult = shouldstop(c, IterationsExecuted, Matrix, CurrentValues)
        if stop
            return stop, foundresult
        end
    end

    return false, false
end

function rayleighquotient(Matrix::AbstractMatrix, X::AbstractVector)
    LinearAlgebra.transpose(X) * Matrix * X
end

function iterationmethod(Matrix::AbstractMatrix,
    StartingValues,
    IterationAction::Function,
    StoppingCriteria::IterativeStoppingCriteria,
    overflowStrategy)

    current = StartingValues
    iteration = 0

    stop, foundresult = shouldstop(StoppingCriteria, iteration, Matrix, current)
    while !stop
        current = IterationAction(Matrix, current, iteration)
        current = overflowStrategy(current)
        iteration += 1
        stop, foundresult = shouldstop(StoppingCriteria, iteration, Matrix, current)
    end

    return iterationresult(Matrix, current, foundresult)
end

function iterationresult(Matrix::AbstractMatrix, V::AbstractVector, FoundResult::Bool)
    EigenResult(EigenEstimates(Matrix, V), FoundResult)
end

function iterationresult(Matrix::AbstractMatrix, Estimates::EigenEstimates, FoundResult::Bool)
    EigenResult(Estimates, FoundResult)
end

for op = (AbstractVector, AbstractVector{<:Integer})
    @eval begin
        function powermethod(Matrix::AbstractMatrix,
            Guess::$op,
            StoppingCriteria::IterativeStoppingCriteria)
        
            updatedGuess = LinearAlgebra.normalize(Guess)
        
            iterationmethod(Matrix,
                updatedGuess,
                (m,c,i) -> m * c,
                StoppingCriteria,
                normalizeStrategy)
        end
    end
end

function inverseiteration(Matrix::AbstractMatrix,
    Shift,
    Guess::AbstractVector,
    StoppingCriteria::IterativeStoppingCriteria)

    iterationmethod(Matrix - LinearAlgebra.UniformScaling(Shift),
        LinearAlgebra.normalize(Guess),
        (m, c, i) -> m \ c,
        StoppingCriteria,
        normalizeStrategy)
end

function rayleighiteration(Matrix::AbstractMatrix,
    Guess::AbstractVector,
    StoppingCriteria::IterativeStoppingCriteria)

    iterationmethod(Matrix,
        EigenEstimates(Matrix, Guess),
        function(m,c,i)
            vecestimate = (m - LinearAlgebra.UniformScaling(c.eigenvalue)) \ c.eigenvector
            valestimate = LinearAlgebra.transpose(vecestimate) * m * vecestimate / (LinearAlgebra.transpose(vecestimate) * vecestimate)
            EigenEstimates(valestimate, vecestimate)
        end,
        StoppingCriteria,
        normalizeStrategy)
end

end