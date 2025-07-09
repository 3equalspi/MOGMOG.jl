function make_batch(mols::Vector{Molecule})
    L = length(mols[1].atom_types)  # vi antar alla har lika många atomer
    B = length(mols)  # batch size

    # Initiera batch-arrayer
    positions = zeros(Float32, 3, L, B)
    atom_types = zeros(Int, L, B)

    for i in 1:B
        positions[:, :, i] .= Float32.(mols[i].positions)
        atom_types[:, i] .= mols[i].atom_types
    end

    return positions, atom_types
end
