#=
@concrete struct MOG
    linear_μ <: Dense         # A layer that predict means
    linear_σ <: Dense         # A layer that predict σ (positive after softplus)
    linear_logw <: Dense      # A layer that predict unnormalized log-weights
end
=#

struct MOG{L}
    layers::L
end

Flux.@layer MOG

function MOG(embed_dim::Int, n_components::Int)
    return MOG((;
        linear_μ = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, n_components)),
        linear_σ = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, n_components, softplus)),
        linear_logw = Chain(StarGLU(embed_dim, 4*embed_dim), Dense(embed_dim, n_components)),
        )
    )
end

function (model::MOG)(axis_embeddings::AbstractArray)
    mog = model.layers
    μ = mog.linear_μ(axis_embeddings)
    σ = mog.linear_σ(axis_embeddings) .+ 0.0005f0
    logw = mog.linear_logw(axis_embeddings)
    logw = logw .- logsumexp(logw; dims=1) # normalize weights to sum to 1
    return μ, σ, logw
end
