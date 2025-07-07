# MOGMOG.jl - A model for molecular embeddings using Mixture of Gaussians (MoG) and Transformers
struct MOGMOGModel
    foot                  # Input encoder (positions + types → embeddings)
    body::Vector{DART}    # Transformer blocks
    mog_head::MoGAxisHead
    atom_head::AtomTypeHead
end

function MOGMOGModel(embed_dim::Int, n_components::Int, vocab_size::Int; depth::Int = 6, n_heads::Int = 4)
    # Bottom encoder: positions + types to embeddings
    foot = encode_prefix  #
    # Transformer body 
    body = [DART(TransformerBlock(embed_dim, n_heads)) for _ in 1:depth]

    # Output heads
    mog_head = MoGAxisHead(embed_dim, n_components)
    atom_head = AtomTypeHead(embed_dim, vocab_size)

    return MOGMOGModel(foot, body, mog_head, atom_head)
end

function (mmm::MOGMOGModel)(positions::Matrix{Float64}, atom_types::Vector{Int})
    # Input encoding
    x = mmm.foot(positions, atom_types)  # (embed_dim, L)

    # Transformer body
    for layer in mmm.body
        x = layer(x)  # (embed_dim, L)
    end

    # Output heads
    μ, σ, logw = mmm.head_x(x)
    logits = mmm.atom_head(x)

    return μ, σ, logw, logits
end
