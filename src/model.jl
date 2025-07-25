# MOGMOG.jl - A model for molecular embeddings using Mixture of Gaussians (MoG) and Transformers
struct MOGMOGModel
    foot::MOGfoot                  # Input encoder (positions + types → embeddings)
    body::Vector{DART}    # Transformer blocks
    mog_head::MoGAxisHead
    atom_head::AtomTypeHead
    climb_head::ClimbHead
    #vocab_size::Int
end

Flux.@layer MOGMOGModel

function MOGMOGModel(embed_dim::Int, n_components::Int, vocab_size::Int; depth::Int = 6, n_heads::Int = 4, max_climb::Int = 10)
    # Bottom encoder
    foot = MOGfoot(embed_dim, vocab_size)
    # Transformer body
    body = [DART(TransformerBlock(embed_dim, n_heads)) for _ in 1:depth]
    # Output heads
    mog_head = MoGAxisHead(embed_dim, n_components)
    atom_head = AtomTypeHead(embed_dim, vocab_size)
    climb_head = ClimbHead(embed_dim, max_climb)
    return MOGMOGModel(foot, body, mog_head, atom_head, climb_head) #vocab_size
end


function (mmm::MOGMOGModel)(positions, atom_types) #atom_types instead
    # Input encoding
    x = mmm.foot(atom_types, positions)  # (embed_dim, L), line 28
    # Transformer body
    for layer in mmm.body
        x = layer(x)  # (embed_dim, L)
    end

    # Output heads
    μ, σ, logw = mmm.mog_head(x[:,2:4,:,:])
    logits = mmm.atom_head(x[:,1:1,:,:])
    climb_logits = mmm.climb_head(x[:,5:5,:,:])

    return μ, σ, logw, logits, climb_logits
end

