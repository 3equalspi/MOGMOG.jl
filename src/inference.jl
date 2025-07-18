atom_type_sample(logits) = logitsample(logits, dims=1)

function mog_sample(μ, σ, logw)
    i = logitsample(logw, dims=1)
    return randn!(similar(logw, size(i))) .* σ[i] .+ μ[i]
end

mog_sample(::typeof(reshape), args...) = dropdims(mog_sample(args...), dims=1)

function sample_next(model::MOGMOGModel, atom_types, positions)
    new_atom_types = [atom_types; similar(atom_types, 1, size(atom_types)[2:end]...) .= 1]
    new_positions = [positions;; selectdim(positions, 2, 1)]
    atom_type_logits, = model(new_atom_types, new_positions)
    new_atom_types[end, :] .= atom_type_sample(atom_type_logits[:, end, :])
    _, μ_x, σ_x, logw_x = model(new_atom_types, new_positions)
    new_positions[1, end, :] .+= mog_sample(reshape, μ_x[:,1,end,:], σ_x[:,1,end,:], logw_x[:,1,end,:])
    _, μ_y, σ_y, logw_y = model(new_atom_types, new_positions)
    new_positions[2, end, :] .+= mog_sample(reshape, μ_y[:,2,end,:], σ_y[:,2,end,:], logw_y[:,2,end,:])
    _, μ_z, σ_z, logw_z = model(new_atom_types, new_positions)
    new_positions[3, end, :] .+= mog_sample(reshape, μ_z[:,3,end,:], σ_z[:,3,end,:], logw_z[:,3,end,:])
    return new_atom_types, new_positions
end

function sample(
    n::Integer,
    model::MOGMOGModel,
    atom_types::AbstractVector{<:Integer}=[1],
    positions::AbstractMatrix{<:AbstractFloat}=fill!(similar(atom_types, Float32, 3, size(atom_types)...), 0),
)
    for _ in 1:n
        atom_types, positions = sample_next(model, atom_types, positions)
        last(atom_types) == 6 && return atom_types[1:end-1], positions[:,1:end-1]
    end
    return atom_types, positions
end
