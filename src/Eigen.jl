module Eigen

function powermethod(Matrix::AbstractMatrix)
    columns = size(Matrix,2)
    guess = rand(columns)

    current = guess
    iterations = 25
    for n in 1:iterations
        current = Matrix * current
    end

    return current
end

end