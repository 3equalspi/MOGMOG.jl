struct MoGAxisHead 
    linear_μ::Dense         # A layer that predict means
    linear_σ::Dense         # A layer that predict σ (positive after softplus)
    linear_logw::Dense      # A layer that predict unnormalized log-weights
end

struct AtomTypeHead
    linear_logits::Dense
end

struct Climbhead
    linear_logits::Dense
end
#h[:,1,:]

function MoGAxisHead(embed_dim::Int, n_components::Int)
    return MoGAxisHead(
        Dense(embed_dim, n_components),
        Dense(embed_dim, n_components, softplus),
        Dense(embed_dim, n_components),
    )
end

function (head::MoGAxisHead)(axis_embeddings::AbstractArray)
    μ = head.linear_μ(axis_embeddings)
    σ = 0.001f0 .+  head.linear_σ(axis_embeddings)
    logw = head.linear_logw(axis_embeddings)
    logw = logw .- logsumexp(logw; dims=1)
    return μ, σ, logw
end


function AtomTypeHead(embed_dim::Int, vocab_size::Int)
    return AtomTypeHead(Dense(embed_dim, vocab_size; bias = false))  #(V, L)
end

function (head::AtomTypeHead)(embeddings::AbstractArray) 
    return head.linear_logits(embeddings)
end

function ClimbHead(embed_dim::Int, max_climb::Int)
    # Predict logits over climb values: 0, 1, ..., max_climb
    return ClimbHead(Dense(embed_dim, max_climb + 1))  # discrete logits
end

function (head::ClimbHead)(embeddings::AbstractMatrix)
    return head.linear_climb(embeddings)  # size: (max_climb+1, L)
end