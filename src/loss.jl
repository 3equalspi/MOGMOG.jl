function logpdf_MOG(x::AbstractArray{Float64}, 
                        μ::AbstractArray{Float64}, 
                        σ::AbstractArray{Float64},
                        logw::AbstractArray{Float64})
    return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end

function loss(model, pos, atoms, atom_dict::Dict{String, Int})

    μ, σ, logw, logits = model(pos[:, 1:end-1])

    displacements = pos[:, 2:end] .- pos[:, 1:end-1]
    displacements = reshape(displacements, 1, size(displacements))

    logp_xyz = logpdf_MOG(displacements, μ, σ, logw)
    loss_xyz = -mean(logp_xyz)

    atom_inds = [atom_dict[a] for a in atoms[2:end]]
    atom_onehot = onehotbatch(atom_inds, 1:length(atom_dict))
    loss_type = logitcrossentropy(logits, atom_onehot)

    return loss_xyz + loss_type
end

export loss
