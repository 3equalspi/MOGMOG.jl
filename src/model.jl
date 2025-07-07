struct MOGMOGModel
    #foot
    body
    head
end

function MOGMOGModel(embed_dim::Int, n_components::Int, vocab_size::Int)

    body = [DART(TransformerBlock(embed_dim, n_heads)) for _ in 1:depth]

    head = MoGAxisHead(embed_dim, n_components)
    atom_head = AtomTypeHead(embed_dim, vocab_size)

    return MOGMOGModel(body, (head)
end

function (mmm::MOGMOGModel)(x::AbstractMatrix)

    for layer in mmm.body
        x = layer(x)
    end

    μ, σ, logw = mmm.head(x)
    logits = mmm.atom_head(x)

    return μ, σ, logw, logits
end