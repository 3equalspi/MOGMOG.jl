@concrete struct MOGencoder
    current_coord_embed
    current_climb_embed
    atom_embed
    index_embed
    next_x_embed
    next_y_embed
    next_z_embed
    token_type_term
end

function MOGencoder(embed_dim::Int, vocab_size::Int, max_climb::Int; rff_dim=128)
    MOGencoder(
        Chain(RandomFourierFeatures(3 => rff_dim, 0.2f0), Dense(rff_dim => embed_dim)), # current coordinates
        Embedding(max_climb => embed_dim), # climb
        Embedding(vocab_size => embed_dim), # atom
        Chain(RandomFourierFeatures(1 => rff_dim, 0.3f0), Dense(rff_dim => embed_dim)), # position
        Chain(RandomFourierFeatures(1 => rff_dim, 0.2f0), Dense(rff_dim => embed_dim)), # next x
        Chain(RandomFourierFeatures(1 => rff_dim, 0.2f0), Dense(rff_dim => embed_dim)), # next y
        Chain(RandomFourierFeatures(1 => rff_dim, 0.2f0), Dense(rff_dim => embed_dim)), # next z
        randn(Float32, embed_dim, 5) .* 0.01f0  # Learnable D x 5 token type embeddings
    )
end

function (encoder::MOGencoder)(
    atom_types::AbstractArray{Int},
    positions::AbstractArray{<:AbstractFloat},
    climbs::AbstractArray{Int},
)
    L = size(atom_types, 1)
    # D x L x B
    atom_embedding = encoder.atom_embed(atom_types[1:L-1, :]) .+
        encoder.current_coord_embed(positions[:, 1:L-1, :]) .+
        encoder.current_climb_embed(1 .+ climbs[1:L-1, :]) .+
        encoder.index_embed(as_dense_on_device((1:L-1)', positions))
    # D x L x B --> D x 5 x L x B
    with_next_atom = atom_embedding + encoder.atom_embed(atom_types[2:L, :])
    with_next_x = with_next_atom + encoder.next_x_embed(positions[1:1, 2:L, :])
    with_next_y = with_next_x + encoder.next_y_embed(positions[2:2, 2:L, :])
    with_next_z = with_next_y + encoder.next_z_embed(positions[3:3, 2:L, :])
    concatenated = vcat(atom_embedding, with_next_atom, with_next_x, with_next_y, with_next_z) # 5D x L x B
    tokens = rearrange(concatenated, ((:d, :k), :l, :b) --> (:d, :k, :l, :b), k=5) # D x 5 x L x B
    tokens = tokens .+ encoder.token_type_term
    return tokens
end
