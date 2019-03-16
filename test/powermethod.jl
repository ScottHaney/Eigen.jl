using Eigen

@testset "Finds the eigenvalue 1 for an identity matrix" begin

    identity = Matrix{Int}(I, 2, 2)
    guess = [1; 0]
    stoppingcriteria = Eigen.ExactlyNIterations(1)

    actual = Eigen.powermethod(identity, guess, stoppingcriteria)
    expected = [1; 0]

    @test actual == expected
end