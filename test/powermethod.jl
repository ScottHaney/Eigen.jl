@testset "Finds the eigenvalue 1 for an identity matrix" begin

    identity = Matrix{Int}(I, 2, 2)
    guess = [1; 0]
    stoppingcriteria = Eigen.ExactlyNIterations(1)

    actual = Eigen.powermethod(identity, guess, stoppingcriteria)
    expected = [1; 0]

    @test actual.eigenvector == expected
    @test actual.isverified == false
end

@testset "Finds the eigenvector for a large eigenvalue using the residual stopping criteria" begin
    matrix = [6 0;0 2]
    guess = [1.0; 2]
    stoppingcriteria = Eigen.Residual(0.1)

    actual = Eigen.powermethod(matrix, guess, stoppingcriteria)
    expected = [1; 0]

    @test LinearAlgebra.norm(actual.eigenvector - expected) <= 0.1
    @test actual.isverified == true
end

@testset "Stops before finding the eigenvector due to a constraint on the maximum number of iterations" begin
    matrix = [6 0; 0 2]
    guess = [1.0; 1]
    stoppingcriteria = Eigen.CompositeStoppingCriteria([Eigen.Residual(0.1), Eigen.ExactlyNIterations(0)])

    actual = Eigen.powermethod(matrix, guess, stoppingcriteria)
    @test LinearAlgebra.norm(actual.eigenvector - LinearAlgebra.normalize(guess)) <= 0.001
    @test actual.isverified == false
end