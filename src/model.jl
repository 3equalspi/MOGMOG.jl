# MOGMOG.jl - A model for molecular embeddings using Mixture of Gaussians (MoG) and Transformers


function pairwise_sqeuclidean(x,y)
    A_sqnorms = sum(abs2, x, dims=2)
    B_sqnorms = sum(abs2, y, dims=1)
    AB_dots = batched_mul(x,y)
    return A_sqnorms .- 2 .* AB_dots .+ B_sqnorms
end

decay(d) = (sign.(d) ./ (1 .+ abs.(d) ./ 1))
function pair_features(positions, anchors, indexes)
    p = positions[:,1:end-1, :]
    a = rearrange(Onion.batched_pairs(==, anchors, indexes), (..) --> (1, ..))
    o = rearrange(p, (:d, :L, :B) --> (:d, 1, :L, :B)) .- rearrange(p, (:d, :L, :B) --> (:d, :L, 1, :B)) #We don't need the other direction on these, because that is just the sign flip
    d = pairwise_sqeuclidean(permutedims(p, (2,1,3)), p)
    e1 = rearrange(1.443f0 .* softplus.(.- (d)), (..) --> (1, ..))
    e2 = rearrange(1.443f0 .* softplus.(.- (d ./ 5)), (..) --> (1, ..))
    e3 = rearrange(1.443f0 .* softplus.(.- (d ./ 15)), (..) --> (1, ..))
    return vcat(a, o, e1, e2, e3)
end


struct MOGMOGModel{L}
    layers::L
end


Flux.@layer MOGMOGModel

function MOGMOGModel(;
    embed_dim=128, mixture_components=32, pair_dim = 32, vocab_size=6, depth=4, heads=4, max_climb=10
)
    
    encoder = MOGencoder(embed_dim, vocab_size, max_climb + 1)
    darts = [DART(TransformerBlock(embed_dim, heads)) for _ in 1:depth]    
    for d in darts
        d.transformer.attention.wo.weight ./= depth
        d.transformer.feed_forward.w2.weight ./= depth
    end
    pair_rff = RandomFourierFeatures(7 => 128, 1.0f0)
    pair_encode = Dense(128 => pair_dim)
    dart_pfs = [Dense(pair_dim => heads) for _ in 1:depth]
    mog_decoder = MOG(embed_dim, mixture_components)
    atom_decoder = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, vocab_size, bias=false))
    climb_decoder = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, max_climb + 1))
    rope = RoPE(embed_dim ÷ heads, 1000)
    control = randn(Float32, embed_dim, 5) 
    return MOGMOGModel((;encoder, pair_rff, pair_encode, darts, dart_pfs, mog_decoder, atom_decoder, climb_decoder, rope, control))
end

function (model::MOGMOGModel)(
    atom_types::AbstractArray{Int},
    positions::AbstractArray{<:AbstractFloat},
    climbs::AbstractArray{Int},
    anchors,
    indexes,
    displacements,
)
    mmm = model.layers
    base_pf = Flux.Zygote.@ignore mmm.pair_rff(pair_features(positions, anchors, indexes))
    pf = mmm.pair_encode(base_pf)
    h = mmm.encoder(atom_types, positions, climbs, anchors, indexes, displacements) # D x L x B. WHAT? # D x 5 x L x B??
    for (di, dart) in enumerate(mmm.darts)
        h = dart(h, mmm.dart_pfs[di](pf); rope=mmm.rope)
    end
    h = h .* swish(mmm.control)
    logits = mmm.atom_decoder(h[:, 1, :, :]) # V x L x B
    μ, σ, logw = mmm.mog_decoder(h[:, 2:4, :, :]) # K x 3 x L x B
    climb_logits = mmm.climb_decoder(h[:, 5, :, :]) # C x L x B
    return logits, μ, σ, logw, climb_logits
end
