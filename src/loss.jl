# loss for mixture of gaussian layer using logpdf
function mog_loss(x, μ, σ, logw)
    return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end

function losses(model, atom_types, positions, atom_mask, coord_mask)
    logits, μ, σ, logw = model(atom_types, positions)

    target_atoms = atom_types[2:end, :]
    disp = positions[:, 2:end, :] .- positions[:, 1:end-1, :]
    disp = reshape(disp, 1, size(disp)...)
    logp_xyz = mog_loss(disp, μ, σ, logw)
    loss_position = -sum(logp_xyz .* reshape(coord_mask, 1, 1, size(coord_mask)...)) / sum(coord_mask)

    atom_onehot = Flux.onehotbatch(target_atoms, 1:size(logits, 1))
    masked_mean(p) = sum(p .* rearrange(atom_mask, (..) --> (1, ..))) / sum(atom_mask)
    loss_atom_type = Flux.logitcrossentropy(logits, atom_onehot; agg=masked_mean)  

    return loss_atom_type, loss_position
end
