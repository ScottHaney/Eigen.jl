@testset "Finds the eigenvector for the largest eigenvalue when that eigenvalue is much larger than the others" begin

    matrix = [5 0;0 1]
    guess = LinearAlgebra.normalize([1;0.5])
    stoppingcriteria = Eigen.stoppingcriteria(0.1, 30)

    actual = Eigen.rayleighiteration(matrix, guess, stoppingcriteria)
    expected = [1;0]

    @test LinearAlgebra.norm(actual.eigenvector - expected) <= 0.1
    @test actual.isvalid == true
end