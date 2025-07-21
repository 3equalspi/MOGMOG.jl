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


function loss_climb1(model, pos, atom_ids, target_atoms, climbs, target_climbs, atom_mask, coord_mask)
    μ, σ, logw, logits, climb_logits = model(pos, atom_ids)
    
    # Coordinate loss
    disp = pos[:, 2:end, :] .- pos[:, 1:end-1, :]
    logp_xyz = logpdf_MOG(disp, μ, σ, logw)
    loss_xyz = -sum(logp_xyz .* coord_mask) / sum(coord_mask)

    # Atom type loss (now computes onehot inside gradient)
    atom_onehot = Flux.onehotbatch(target_atoms, 1:size(logits, 1))
    loss_type = Flux.logitcrossentropy(logits, atom_onehot; 
                                     agg=x -> sum(x .* atom_mask)/sum(atom_mask))

    # Climb prediction loss
    climb_onehot = Flux.onehotbatch(target_climbs, 0:size(climb_logits, 1)-1)
    loss_climb = Flux.logitcrossentropy(climb_logits, climb_onehot;
                                      agg=x -> sum(x .* atom_mask)/sum(atom_mask))

    return loss_xyz + loss_type + loss_climb
end

function loss_climb(model, pos, atom_ids, target_atoms, climbs, target_climbs, atom_mask, coord_mask)
    # Forward pass
    μ, σ, logw, logits, climb_logits = model(pos, atom_ids) #line 54
    
    # Coordinate loss
    disp = pos[:,2:end,:] .- pos[:,1:end-1,:]
    logp_xyz = logpdf_MOG(disp, μ, σ, logw)
    loss_xyz = -sum(logp_xyz .* coord_mask) / sum(coord_mask)

    # Atom type loss
    atom_onehot = Flux.onehotbatch(target_atoms, 1:model.vocab_size)
    loss_atom = Flux.logitcrossentropy(
        dropdims(logits, dims=2),  # Remove singleton dimension
        atom_onehot,
        agg=x->sum(x.*atom_mask)/sum(atom_mask)
    )

    # Climb prediction loss
    climb_onehot = Flux.onehotbatch(target_climbs, 0:model.climb_head.linear_climb.out-1)
    loss_climb = Flux.logitcrossentropy(
        dropdims(climb_logits, dims=2),
        climb_onehot,
        agg=x->sum(x.*atom_mask)/sum(atom_mask)
    )

    return loss_xyz + loss_atom + loss_climb
end