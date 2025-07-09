function logpdf_MOG(x::AbstractArray{<:AbstractFloat}, 
                        μ::AbstractArray{<:AbstractFloat}, 
                        σ::AbstractArray{<:AbstractFloat},
                        logw::AbstractArray{<:AbstractFloat})
    return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end

# 11.586529630541413
export loss

function loss(model, atom_ids, pos, atom_mask, coord_mask)
    target_atoms = atom_ids[2:end, :]
 
    μ, σ, logw, logits = model(pos, atom_ids)
    disp = pos[:, 2:end, :] .- pos[:, 1:end-1, :]
    disp = reshape(disp, 1, size(disp)...)
    logp_xyz = logpdf_MOG(disp, μ, σ, logw)
    loss_xyz = -sum(logp_xyz .* reshape(coord_mask, 1, 1, size(coord_mask)...)) / sum(coord_mask)
 
    atom_onehot = Flux.onehotbatch(target_atoms, 1:size(logits, 1))

    atom_mask = reshape(atom_mask, 1, size(atom_mask)...)
    masked_mean(p) = sum(p .* atom_mask) / sum(atom_mask)
 
    loss_type = Flux.logitcrossentropy(dropdims(logits, dims=2), atom_onehot; agg=masked_mean)
 
    return loss_xyz + loss_type
end