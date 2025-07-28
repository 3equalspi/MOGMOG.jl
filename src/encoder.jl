@concrete struct MOGencoder
    current_coord_embed
    current_climb_embed
    atom_embed
    next_atom_embed
    index_embed
    next_x_embed
    next_y_embed
    next_z_embed
    token_type_term
end

Flux.@layer MOGencoder

function MOGencoder(embed_dim::Int, vocab_size::Int, max_climb::Int; rff_dim=128)
    MOGencoder(
        Chain(RandomFourierFeatures(3 => rff_dim, 1.0f0), Dense(rff_dim => embed_dim)), # current coordinates
        Embedding(max_climb => embed_dim), # climb
        Embedding(vocab_size => embed_dim), # atom
        Embedding(vocab_size => embed_dim), # nextatom
        Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => embed_dim)), # position
        Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => embed_dim)), # next x
        Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => embed_dim)), # next y
        Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => embed_dim)), # next z
        randn(Float32, embed_dim, 5) .* 0.1f0  # Learnable D x 5 token type embeddings
    )
end

function (encoder::MOGencoder)(
    atom_types::AbstractArray{Int},
    positions::AbstractArray{<:AbstractFloat},
    climbs::AbstractArray{Int},
    anchors,
    indexes,
    displacements,
)
    L = size(atom_types, 1)
    # D x L x B
    index_embedding = encoder.index_embed(reshape(indexes, 1, size(indexes)...) ./ 10f0)
    atom_embedding = encoder.atom_embed(atom_types[1:L-1, :]) .+
        encoder.current_coord_embed(positions[:, 1:L-1, :]) .+
        encoder.current_climb_embed(1 .+ climbs[1:L-1, :]) .+
        index_embedding
    # D x L x B --> D x 5 x L x B
    with_next_atom = index_embedding .+ encoder.next_atom_embed(atom_types[2:L, :])
    with_next_x = index_embedding .+ encoder.next_x_embed(displacements[1:1, :, :]) 
    with_next_y = index_embedding .+ encoder.next_y_embed(displacements[2:2, :, :])
    with_next_z = index_embedding .+ encoder.next_z_embed(displacements[3:3, :, :])
    concatenated = vcat(atom_embedding, with_next_atom, with_next_x, with_next_y, with_next_z) # 5D x L x B
    tokens = rearrange(concatenated, ((:d, :k), :l, :b) --> (:d, :k, :l, :b), k=5) # D x 5 x L x B
    tokens = tokens .+ encoder.token_type_term
    return tokens
end
