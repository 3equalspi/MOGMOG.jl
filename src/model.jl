# MOGMOG.jl - A model for molecular embeddings using Mixture of Gaussians (MoG) and Transformers
struct MOGMOGModel
    foot                  # Input encoder (positions + types → embeddings)
    body::Vector{DART}    # Transformer blocks
    head_x::MoGAxisHead
    head_y::MoGAxisHead
    head_z::MoGAxisHead
    atom_head::AtomTypeHead
end

function MOGMOGModel(embed_dim::Int, n_components::Int, vocab_size::Int; depth::Int = 6, n_heads::Int = 4)
    # Bottom encoder: positions + types to embeddings
    foot = encode_prefix  #
    # Transformer body 
    body = [DART(TransformerBlock(embed_dim, n_heads)) for _ in 1:depth]

    # Output heads
    head_x = MoGAxisHead(embed_dim, n_components)
    head_y = MoGAxisHead(embed_dim, n_components)
    head_z = MoGAxisHead(embed_dim, n_components)
    atom_head = AtomTypeHead(embed_dim, vocab_size)

    return MOGMOGModel(foot, body, head_x, head_y, head_z, atom_head)
end

function (mmm::MOGMOGModel)(positions::Matrix{Float64}, atom_types::Vector{Int})
    # Input encoding
    x = mmm.foot(positions, atom_types)  # (embed_dim, L)

    # Transformer body
    for layer in mmm.body
        x = layer(x)  # (embed_dim, L)
    end

    # Output heads
    μx, σx, logwx = mmm.head_x(x)
    μy, σy, logwy = mmm.head_y(x)
    μz, σz, logwz = mmm.head_z(x)
    logits = mmm.atom_head(x)

    return μx, σx, logwx, μy, σy, logwy, μz, σz, logwz, logits
end
