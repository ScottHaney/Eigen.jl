@testset "Finds the eigenvector for the largest eigenvalue when that eigenvalue is much larger than the others" begin

    matrix = [1 0;0 1]
    guess = LinearAlgebra.normalize([0.9;0.1])
    stoppingcriteria = Eigen.stoppingcriteria(100)

    actual = Eigen.rayleighiteration(matrix, guess, stoppingcriteria)
    expected = [1;0]

    @test LinearAlgebra.norm(actual.eigenvector - expected) <= 0.1
end