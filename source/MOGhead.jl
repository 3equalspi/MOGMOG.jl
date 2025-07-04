struct MoGAxisHead 
    linear_μ::Dense         # A layer that predict means
    linear_σ::Dense         # A layer that predict σ (positive after softplus)
    linear_logw::Dense      # A layer that predict unnormalized log-weights
end

function MoGAxisHead(embed_dim::Int, n_components::Int)
    return MoGAxisHead(
        Dense(embed_dim, n_components),
        Dense(embed_dim, n_components, softplus),
        Dense(embed_dim, n_components),
    )
end

function (head::MoGAxisHead)(axis_embeddings::AbstractMatrix)
    μ = head.linear_μ(axis_embeddings)
    σ = head.linear_σ(axis_embeddings)
    logw = head.linear_logw(axis_embeddings)
    logw = logw .- logsumexp(logw; dims=1)

    return μ, σ, logw
end
