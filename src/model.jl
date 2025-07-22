# MOGMOG.jl - A model for molecular embeddings using Mixture of Gaussians (MoG) and Transformers
@concrete struct MOGMOGModel
    encoder <: MOGencoder                 # Input encoder (positions + types → embeddings)
    norm
    darts <: Tuple{Vararg{<:DART}}    # Transformer blocks
    mog_decoder <: MOG
    atom_decoder <: Dense
    climb_decoder <: Dense
end

Flux.@layer MOGMOGModel

function MOGMOGModel(;
    embed_dim=128, mixture_components=16, vocab_size=6, depth=4, heads=4, max_climb=10
)
    # Input encoder
    encoder = MOGencoder(embed_dim, vocab_size)

    # Doubly Auto-Regressive Transformer stack
    darts = ntuple(i -> DART(TransformerBlock(embed_dim, heads)), depth)
    for d in darts
        d.transformer.attention.wo.weight ./= 10
        d.transformer.feed_forward.w2.weight ./= 10
    end

    # Decoders
    mog_decoder = MOG(embed_dim, mixture_components)
    atom_decoder = Dense(embed_dim, vocab_size, bias=false)
    climb_decoder = Dense(embed_dim, max_climb + 1)

    return MOGMOGModel(encoder, LayerNorm(embed_dim), darts, mog_decoder, atom_decoder, climb_decoder)
end

function (mmm::MOGMOGModel)(
    atom_types::AbstractArray{Int},
    positions::AbstractArray{<:AbstractFloat},
    climbs::AbstractArray{Int}
)
    h = mmm.encoder(atom_types, positions, climbs) # D x L x B
    h = mmm.norm(h)

    for dart in mmm.darts
        h = dart(h)
    end

    logits = mmm.atom_decoder(h[:, 1, :, :]) # V x L x B
    μ, σ, logw = mmm.mog_decoder(h[:, 2:4, :, :]) # K x 3 x L x B
    climb_logits = mmm.climb_decoder(h[:, 5, :, :]) # C x L x B

    return logits, μ, σ, logw, climb_logits
end
