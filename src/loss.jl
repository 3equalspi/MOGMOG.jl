function logpdf_MOG(x::AbstractArray{<:AbstractFloat}, 
                        μ::AbstractArray{<:AbstractFloat}, 
                        σ::AbstractArray{<:AbstractFloat},
                        logw::AbstractArray{<:AbstractFloat})
    @show size(x)
    @show size(μ)
    @show size(σ)
    @show size(logw)
    return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end

# 11.586529630541413
function loss(model, pos, atoms, vocab_size)

    μ, σ, logw, logits = model(pos, atoms)

    displacements = pos[:, 2:end] .- pos[:, 1:end-1]
    displacements = reshape(displacements, 1, size(displacements)...)

    logp_xyz = logpdf_MOG(
        displacements,
        μ,
        σ,
        logw)
    loss_xyz = -mean(logp_xyz)

    atom_onehot = Flux.onehotbatch(atoms[2:end], 1:vocab_size)
    loss_type = Flux.logitcrossentropy(
        rearrange(logits, (:v, 1, :l_1, 1) --> (:v, :l_1)),
        atom_onehot) # vocab_size x L+1

    return loss_xyz + loss_type
end

export loss
