@concrete struct MOG
    linear_μ <: Dense         # A layer that predict means
    linear_σ <: Dense         # A layer that predict σ (positive after softplus)
    linear_logw <: Dense      # A layer that predict unnormalized log-weights
end

function MOG(embed_dim::Int, n_components::Int)
    return MOG(
        Dense(embed_dim, n_components),
        Dense(embed_dim, n_components, softplus),
        Dense(embed_dim, n_components),
    )
end

function (mog::MOG)(axis_embeddings::AbstractArray)
    μ = mog.linear_μ(axis_embeddings)
    σ = mog.linear_σ(axis_embeddings) .+ 0.003f0
    logw = mog.linear_logw(axis_embeddings)
    logw = logw .- logsumexp(logw; dims=1) # normalize weights to sum to 1
    return μ, σ, logw
end
