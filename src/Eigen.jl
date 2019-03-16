module Eigen

function powermethod(Matrix::AbstractMatrix, Guess::AbstractMatrix)
    current = Guess
    iterations = 25
    for n in 1:iterations
        current = Matrix * current
    end

    return current
end

end