atom_type_sample(logits) = logitsample(Top_p(0.95f0)(logits |> vec), dims=1)

function mog_sample(μ, σ, logw)
    i = logitsample(logw, dims=1)
    return randn!(similar(logw, size(i))) .* σ[i] .+ μ[i]
end

mog_sample(::typeof(reshape), args...) = rearrange(mog_sample(args...), (1, ..) --> (..))

function nucleus_sample(draws, probs; p = 0.9)
    cap = Int(round(length(probs)*p))
    keeps = sortperm(probs, rev = true)[1:cap]
    return rand(draws[keeps])
end

function nucleus_mog_sample(μ, σ, logw; p = 0.95, n = 200)
    samps = stack([mog_sample(μ, σ, logw) for i in 1:n])
    probs = vec(exp.(mog_logpdf(samps, μ, σ, logw)))
    return nucleus_sample(samps, probs, p = p) 
end

function sample_next(model::MOGMOGModel, atom_types, positions, climbs)
    #@show atom_types, positions, climbs
    new_atom_types = [atom_types; 1]
    new_climbs = [climbs; 0]
    from = first.(climbs_to_pairs(new_climbs[2:end]))[end] #<- sus
    @show from
    new_positions = [positions;; selectdim(positions, 2, from)]
    atom_type_logits, = model(new_atom_types, new_positions, new_climbs)
    new_atom_types[end, :] .= atom_type_sample(atom_type_logits[:, end, :])
    _, μ_x, σ_x, logw_x = model(new_atom_types, new_positions, new_climbs)
    new_positions[1, end, :] .+= nucleus_mog_sample(μ_x[:,1,end,1], σ_x[:,1,end,1], logw_x[:,1,end,1], p = 0.9, n = 400)
    _, μ_y, σ_y, logw_y = model(new_atom_types, new_positions, new_climbs)
    new_positions[2, end, :] .+= nucleus_mog_sample(μ_y[:,2,end,1], σ_y[:,2,end,1], logw_y[:,2,end,1], p = 0.9, n = 400)
    _, μ_z, σ_z, logw_z = model(new_atom_types, new_positions, new_climbs)
    new_positions[3, end, :] .+= nucleus_mog_sample(μ_z[:,3,end,1], σ_z[:,3,end,1], logw_z[:,3,end,1], p = 0.9, n = 400)
    _, _, _, _, climb_logits = model(new_atom_types, new_positions, new_climbs)
    new_climbs[end, :] .= logitsample(Top_p(0.999f0)(climb_logits[:, end, 1]), dims=1) .- 1
    return new_atom_types, new_positions, new_climbs
end

function sample(
    n::Integer,
    model::MOGMOGModel,
    atom_types::AbstractVector{<:Integer}=[1],
    positions::AbstractMatrix{<:AbstractFloat}=fill!(similar(atom_types, Float32, 3, size(atom_types)...), 0),
    climbs::AbstractVector{<:Integer}=Int[0],
)
    for i in 1:n
        atom_types, positions, climbs = sample_next(model, atom_types, positions, climbs)
        if last(atom_types) == 6
            display(climbs)
            return atom_types[1:end-1], positions[:,1:end-1]
        end
    end
    return atom_types, positions
end




















#=
function sample_next(model::MOGMOGModel, atom_types, positions, climbs, memory, t, n)
    for i in 1:n
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
        push!(get!(memory, t, []), (new_atom_types[end], new_positions[:,end], new_climbs[end]))
    end
    return new_atom_types, new_positions, new_climbs
end

function sample(
    n::Integer,
    model::MOGMOGModel,
    atom_types::AbstractVector{<:Integer}=[1],
    positions::AbstractMatrix{<:AbstractFloat}=fill!(similar(atom_types, Float32, 3, size(atom_types)...), 0),
    climbs::AbstractVector{<:Integer}=Int[],
)
    memory = Dict()
    for i in 1:n
        atom_types, positions, climbs = sample_next(model, atom_types, positions, climbs, memory, i, 100)
        last(atom_types) == 6 && return atom_types[1:end-1], positions[:,1:end-1]
    end
    return atom_types, positions, memory
end
=#