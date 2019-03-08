using Eigen

@testset "Finds the eigenvalue 1 for an identity matrix" begin

    @test Eigen.EigenTest(2) == 4

end