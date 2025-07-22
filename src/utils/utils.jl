include("climbs.jl")

centered(X::AbstractArray{<:Number}) = X .- mean(X, dims=2)

# apply random rigid transformation to a molecule
# with translation standard deviation σ
function apply_random_rigid(X::AbstractArray{T}, σ::T=one(T)) where T<:Number
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


function pad_and_batch(molecules, vocab_dict, pad_token="STOP"; center=false, random_rigid=false)
    max_len = maximum(m -> length(m.atom_types), molecules, init=0) + 1
    B = length(molecules)
    PAD = vocab_dict[pad_token]
    atom_types = fill(PAD, max_len, B)
    positions = zeros(Float32, 3, max_len, B)
    displacements = zeros(Float32, 3, max_len - 1, B)
    climbs = zeros(Int, max_len - 1, B)
    atom_mask = zeros(Float32, max_len - 1, B)
    coord_mask = zeros(Float32, max_len - 1, B)
    
    for (i, mol) in enumerate(molecules)
        L = length(mol.atom_types)
        for j in 1:L
            atom_types[j, i] = get(vocab_dict, mol.atom_types[j], PAD)
        end
        positions[:, 1:L, i] = mol.positions[:, 1:L]
        displacements[:, 1:L-1, i] = mol.positions[:, 2:end] .- mol.positions[:, first.(mol.edges)]
        climbs[1:L-1, i] = mol.climbs
        coord_mask[1:L-1, i] .= 1
        atom_mask[1:L, i] .= 1
    end

    center && (positions = centered(positions))
    random_rigid && (positions = apply_random_rigid(positions))

    return (; atom_types, positions, displacements, climbs, atom_mask, coord_mask)
end


as_dense_on_device(x, array::DenseArray) = similar(array, size(x)) .= x
@non_differentiable as_dense_on_device(::Any...)


function to_xyz(elements::AbstractVector, positions::AbstractMatrix)
    join("$e $x $y $z\n" for (e, (x, y, z)) in zip(elements, eachcol(positions)))
end