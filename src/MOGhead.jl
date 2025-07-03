struct MoGAxisHead
    linear_μ::Dense         # Predict means
    linear_σ::Dense      # Predict σ (positive after softplus)
    linear_logw::Dense      # Predict unnormalized log-weights
end

function MoGAxisHead(embed_dim::Int, n_components::Int)
    return MoGAxisHead(
        Dense(embed_dim, n_components),      # μ_k
        Dense(embed_dim, n_components, softplus),      # σ_k
        Dense(embed_dim, n_components),      # log w_k
    )
end

function (head::MoGAxisHead)(axis_embeddings::AbstractMatrix)
    # axis_embeddings: (embed_dim, L)

    μ = head.linear_μ(axis_embeddings)
    σ = head.linear_σ(axis_embeddings) 
    logw = head.linear_logw(axis_embeddings)     
    logw = logw .- logsumexp(logw; dims=1)       # Normalize log w over K

    return μ, σ, logw
end