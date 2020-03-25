@testset "Finds a trivial arnoldi decomposition" begin

    matrix = [1 1 1;1 1 1;0 1 1];
    guess = [1;0;0]
    
    expected = [1 1 1;1 1 1;0 1 1]
    actual = Eigen.arnoldi(matrix, guess, 3)

    @test expected == actual
end