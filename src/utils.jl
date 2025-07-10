struct Molecule
    atoms::Vector{String}
    positions::Matrix{Float64}
end

Base.length(mol::Molecule) = length(mol.atoms)

centroid(X::AbstractArray{<:Number}) = mean()

# apply random rigid transformation to a molecule
# with translation standard deviation σ
function apply_random_rigid(X::AbstractArray{T}, σ::T=zero(T)) where T<:Number
    @assert size(X, 1) == 3

    Q, _ = qr(randn!(similar(X, 3, 3)))
    if det(Q) < 0
        Q[:,1] .*= -1
    end
    R = Q
    t = randn!(similar(X, 3)) * σ
    
    X′ = reshape(X, 3, :)
    Y′ = R * X′ .+ t
    Y = reshape(Y′, size(X))
    return Y
end

export transform_molecule


function pad_and_batch(molecules::Vector{Molecule}, vocab_dict, pad_token="STOP"; random_rigid=false)
    max_len = maximum(length, molecules, init=0) + 1
    B = length(molecules)
    PAD = vocab_dict[pad_token]
    atom_types = fill(PAD, max_len, B)
    positions = zeros(Float32, 3, max_len, B)
    atom_type_mask = zeros(Float32, max_len - 1, B)
    coordinate_mask = zeros(Float32, max_len - 1, B)
    
    for (i, mol) in enumerate(molecules)
        L = length(mol)
        for j in 1:L
            atom_types[j, i] = get(vocab_dict, mol.atoms[j], PAD)
        end
        positions[:, 1:L, i] = mol.positions[:, 1:L]
        coordinate_mask[1:L-1, i] .= 1.0
        atom_type_mask[1:L, i] .= 1.0
    end

    if random_rigid
        positions = apply_random_rigid(positions)
    end

    return atom_types, positions, atom_type_mask, coordinate_mask
end
