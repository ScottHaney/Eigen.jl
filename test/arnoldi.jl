@testset "Finds a trivial complete arnoldi decomposition" begin

    matrix = [1 1 1;1 1 1;0 1 1];
    guess = [1;0;0]
    
    expected = [1 1 1;1 1 1;0 1 1]
    actual = Eigen.arnoldi(matrix, guess, 3)

    @test expected == actual.H
end

@testset "Finds a trivial partial arnoldi decomposition" begin

    matrix = [1 1 1;1 1 1;0 1 1];
    guess = [1;0;0]
    
    expected = [1 1;1 1;0 1]
    actual = Eigen.arnoldi(matrix, guess, 2)

    @test expected == actual.H
end

@testset "Stops the decomposition if an invariant subspace is found" begin

    matrix = [1 0 0;0 1 0;0 0 1]
    guess = [1;0;0]
    
    expected = 1
    actual = Eigen.arnoldi(matrix, guess, 2)

    @test expected == actual.numColumns
end