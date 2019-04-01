@testset "Finds the eigenvalue 5 when choosing an estimate near it" begin

    matrix = [20 0;0 5]
    guessshift = 4.8
    guess = [1;1]
    stoppingcriteria = Eigen.stoppingcriteria(0.1, 30)

    actual = Eigen.inverseiteration(matrix, guessshift, guess, stoppingcriteria)
    expected = [0; 1]

    @test LinearAlgebra.norm(actual.eigenvector - expected) <= 0.1
    @test actual.isvalid == true
end