using MOGMOG
using Test

@testset "MOGMOG.jl" begin

    @testset "climb" begin
        @test MOGMOG.smiles_to_climbs("C(C)C") == [1, 0]
        @test MOGMOG.smiles_to_climbs("C(C(C))") == [0, 2]
        @test MOGMOG.smiles_to_climbs("C(C(CC))") == [0, 0, 3]
        @test MOGMOG.smiles_to_climbs("C(C(C))C") == [0, 2, 0]
        @test MOGMOG.smiles_to_climbs("CC(C(C)C)(CC(C)CC)C(C)C") == [0, 0, 1, 2, 0, 0, 1, 0, 4, 0, 1, 0]
    end

end
