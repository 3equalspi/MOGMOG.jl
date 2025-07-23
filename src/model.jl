# MOGMOG.jl - A model for molecular embeddings using Mixture of Gaussians (MoG) and Transformers

struct MOGMOGModel{L}
    layers::L
end

Flux.@layer MOGMOGModel

function MOGMOGModel(;
    embed_dim=128, mixture_components=16, vocab_size=6, depth=4, heads=4, max_climb=10
)
    encoder = MOGencoder(embed_dim, vocab_size, max_climb + 1)
    darts = [DART(TransformerBlock(embed_dim, heads)) for _ in 1:depth]
    for d in darts
        d.transformer.attention.wo.weight ./= 10
        d.transformer.feed_forward.w2.weight ./= 10
    end
    mog_decoder = MOG(embed_dim, mixture_components)
    atom_decoder = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, vocab_size, bias=false))
    climb_decoder = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, max_climb + 1))
    rope = RoPE(embed_dim ÷ heads, 1000)
    control = randn(Float32, embed_dim, 5, 1, 1) 
    return MOGMOGModel((;encoder, darts, mog_decoder, atom_decoder, climb_decoder, rope, control))
end

function (model::MOGMOGModel)(
    atom_types::AbstractArray{Int},
    positions::AbstractArray{<:AbstractFloat},
    climbs::AbstractArray{Int}
)
    mmm = model.layers
    h = mmm.encoder(atom_types, positions, climbs) # D x L x B. WHAT? # D x 5 x L x B??
    for dart in mmm.darts
        h = dart(h; rope=mmm.rope)
    end
    h = h .* swish(mmm.control)
    logits = mmm.atom_decoder(h[:, 1, :, :]) # V x L x B
    μ, σ, logw = mmm.mog_decoder(h[:, 2:4, :, :]) # K x 3 x L x B
    climb_logits = mmm.climb_decoder(h[:, 5, :, :]) # C x L x B
    return logits, μ, σ, logw, climb_logits
end
