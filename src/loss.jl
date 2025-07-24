# loss for mixture of gaussian layer using logpdf
function mog_logpdf(x, μ, σ, logw)
    return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end


function losses(model, batch)
    atom_logits, μ, σ, logw, climb_logits = model(
        batch.atom_types, batch.positions, batch.climbs, batch.anchors, batch.indexes)
    atom_onehot = Flux.onehotbatch(batch.atom_types[2:end, :], 1:size(atom_logits, 1))
    atom_masked_mean(p) = sum(p .* rearrange(batch.atom_mask, (..) --> (1, ..))) / sum(batch.atom_mask)
    loss_atom_type = Flux.logitcrossentropy(atom_logits, atom_onehot; agg=atom_masked_mean)
    logp_xyz = mog_logpdf(rearrange(batch.displacements, (..) --> (1, ..)), μ, σ, logw)
    #=
    if rand() < 0.1
        threeloss = -sum(logp_xyz .* rearrange(batch.coord_mask, (..) --> (1, 1, ..)), dims = (3,4))[:] / sum(batch.coord_mask)
        @show threeloss
    end
    =#
    loss_position = -sum(logp_xyz .* rearrange(batch.coord_mask, (..) --> (1, 1, ..))) / sum(batch.coord_mask)
    #This indexing is an ugly attempt at a quickfix, and needs careful checking, and maybe a re-work to be clearer.
    climb_onehot = Flux.onehotbatch(batch.climbs[2:end, :], 0:size(climb_logits, 1)-1)
    climb_masked_mean(p) = sum(p .* rearrange(batch.coord_mask[1:end-1, :], (..) --> (1, ..))) / sum(batch.coord_mask[1:end-1, :])
    loss_climb = Flux.logitcrossentropy(climb_logits[:,1:end-1, :], climb_onehot; agg=climb_masked_mean)

    return loss_atom_type, loss_position, loss_climb
end
