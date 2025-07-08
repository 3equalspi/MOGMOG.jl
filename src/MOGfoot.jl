using Einops

struct MOGfoot
    current_coord_embed
    atom_embed
    position_embed
    next_x_embed
    next_y_embed
end

function MOGfoot(embed_dim::Int, vocab_size::Int)
    MOGfoot(
        Chain(RandomFourierFeatures(3 => embed_dim, 0.1f0), Dense(embed_dim => embed_dim)), # current coordinates
        Embedding(vocab_size => embed_dim), # atom
        Chain(RandomFourierFeatures(1 => embed_dim, 0.5f0), Dense(embed_dim => embed_dim)), # position
        Chain(RandomFourierFeatures(1 => embed_dim, 0.1f0), Dense(embed_dim => embed_dim)), # next x
        Chain(RandomFourierFeatures(1 => embed_dim, 0.1f0), Dense(embed_dim => embed_dim)), # next y
    )
end

# atom types: L x B
# [1, 5, 3, 7, 8]
# coordinates: 3 x L x B
# 3×5 Matrix{Float64}:
#  0.163998   0.324916   0.866443  0.0771369  0.0660904
#  0.306506   0.0206321  0.417595  0.76132    0.203126
#  0.0464874  0.290774   0.102366  0.206444   0.338374
function (foot::MOGfoot)(atom_types::AbstractArray{Int}, coordinates::AbstractArray{<:AbstractFloat})
    L = length(atom_types)
    # D x L x B
    atom_embedding = foot.atom_embed(atom_types[1:L-1, :]) .+
        foot.current_coord_embed(coordinates[:,1:L-1, :]) .+
        foot.position_embed(reshape(Float32.(1:L-1), 1, L-1))
    # D x L x B --> D x 4 x L x B
    with_next_atom = atom_embedding + foot.atom_embed(atom_types[2:L, :])
    with_next_x = with_next_atom + foot.next_x_embed(coordinates[1:1, 2:L, :])
    with_next_y = with_next_x + foot.next_y_embed(coordinates[2:2, 2:L, :])
    concatenated = vcat(atom_embedding, with_next_atom, with_next_x, with_next_y) # 4D x L x B
    return rearrange(concatenated, einops"(d k) l b -> d k l b", k=4)
end
