atom_type_sample(logits) = logitsample(logits, dims=1)

function mog_sample(μ, σ, logw)
    i = logitsample(logw, dims=1)
    return randn!(similar(logw, size(i))) .* σ[i] .+ μ[i]
end

mog_sample(::typeof(reshape), args...) = rearrange(mog_sample(args...), (1, ..) --> (..))

function sample_next(model::MOGMOGModel, atom_types, positions, climbs)
    new_atom_types = [atom_types; 1]
    new_climbs = [climbs; 0]
    from = first.(climbs_to_pairs(new_climbs))[end]
    new_positions = [positions;; selectdim(positions, 2, from)]
    atom_type_logits, = model(new_atom_types, new_positions, new_climbs)
    new_atom_types[end, :] .= atom_type_sample(atom_type_logits[:, end, :])
    _, μ_x, σ_x, logw_x = model(new_atom_types, new_positions, new_climbs)
    new_positions[1, end, :] .+= mog_sample(reshape, μ_x[:,1,end,:], σ_x[:,1,end,:], logw_x[:,1,end,:])
    _, μ_y, σ_y, logw_y = model(new_atom_types, new_positions, new_climbs)
    new_positions[2, end, :] .+= mog_sample(reshape, μ_y[:,2,end,:], σ_y[:,2,end,:], logw_y[:,2,end,:])
    _, μ_z, σ_z, logw_z = model(new_atom_types, new_positions, new_climbs)
    new_positions[3, end, :] .+= mog_sample(reshape, μ_z[:,3,end,:], σ_z[:,3,end,:], logw_z[:,3,end,:])
    _, _, _, _, climb_logits = model(new_atom_types, new_positions, new_climbs)
    new_climbs[end, :] .= logitsample(climb_logits[:, end, :], dims=1) .- 1
    return new_atom_types, new_positions, new_climbs
end

function sample(
    n::Integer,
    model::MOGMOGModel,
    atom_types::AbstractVector{<:Integer}=[1],
    positions::AbstractMatrix{<:AbstractFloat}=fill!(similar(atom_types, Float32, 3, size(atom_types)...), 0),
    climbs::AbstractVector{<:Integer}=Int[],
)
    for i in 1:n
        atom_types, positions, climbs = sample_next(model, atom_types, positions, climbs)
        last(atom_types) == 6 && return atom_types[1:end-1], positions[:,1:end-1]
    end
    return atom_types, positions
end
