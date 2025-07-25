using Pkg
Pkg.activate(".")
using Revise
#Pkg.add(["Flux", "LearningSchedules", "Random", "JLD2", "Serialization", "PeriodicTable", "Plots", "CannotWaitForTheseOptimisers", "Einops", "CUDA", "cuDNN"])
Pkg.develop(path="../")



using MOGMOG

using Flux
using LearningSchedules
using Random
using JLD2
using Serialization
using PeriodicTable
using Plots
using CannotWaitForTheseOptimisers

GPUnum = 0
ENV["CUDA_VISIBLE_DEVICES"] = GPUnum

using CUDA

Random.seed!(0)

molecules = deserialize("qm9.jls")

function prepare(mol)
    I = findall(!=('H'), mol.elements)
    atom_types = string.(mol.elements[I])
    positions = mol.coords[:,I]
    pre_climbs = MOGMOG.smiles_to_climbs(mol.smiles)
    edges = MOGMOG.climbs_to_pairs(pre_climbs)
    climbs = vcat([0],pre_climbs) #<-pre-padding climb with a zero, which will be the input to the first position (only after building the edges)
    return (; atom_types, positions, climbs, edges)
end
data = prepare.(molecules)

unique_atoms = unique(vcat((mol.atom_types for mol in data)...))
vocabulary = sort([unique_atoms; "H"; "STOP"])
vocab_dict = Dict(name => i for (i, name) in enumerate(vocabulary))
reverse_vocab_dict = Dict(i => name for (i, name) in enumerate(vocabulary))


#=
function sdpa(xq::AbstractArray{T}, xk::AbstractArray{T}, xv::AbstractArray{T}, mask::AbstractArray{T, 3}) where T
    A = softmax(batched_mul(batched_transpose(xk), xq) / sqrt(T(size(xq, 1))) .+ mask, dims=1)
    return batched_mul(xv, A)
end
=#
@eval MOGMOG.Onion begin
    function (dart::DART)(x::AbstractArray; mask=:causal, rope=nothing)
        h = rearrange(x, (:d, :K, :L, ..) --> (:d, (:K, :L), ..))
        mask === :causal && (mask = causal_mask(h))
        L = size(h, 2)
        return reshape(dart.transformer(h; mask, rope = rope[1:L]), size(x))
    end
    function (dart::DART)(x::AbstractArray, pair_attlogit_bias::AbstractArray; mask=:causal, rope=nothing)
        h = rearrange(x, (:d, :K, :L, ..) --> (:d, (:K, :L), ..))
        k = size(x, 2)
        dpf = repeat(pair_attlogit_bias, (:k, :l1, :l2, :b) --> ((k, :l1), (k, :l2), (:k, :b))) 
        mask === :causal && (dpf = dpf .+ rearrange(causal_mask(h), (..) --> (.., 1)))
        return reshape(dart.transformer(h; mask = dpf, rope = rope[1:size(h, 2)]), size(x))
    end
end

ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# Hyperparameters
embed_dim = 256
mixture_components = 32
vocab_size = length(vocab_dict)
depth = 10
heads = 8
batchsize = 64
nbatches = 2000
nepochs = 100

# Initialize model and optimizer
model = MOGMOGModel(; embed_dim, mixture_components, vocab_size, depth, heads) |> gpu;
scheduler = burnin_learning_schedule(0.000001f0, 0.001f0, 1.005f0, 0.9999f0)
opt_state = Flux.setup(Muon(eta = scheduler.lr), model)



#scheduler = linear_decay_schedule(0.000153f0, 0.0000000001f0, 9800) 

# Training loop
all_losses = Float32[]

for epoch in 1:nepochs
    println("Epoch $epoch")
    shuffled = shuffle(data)
    for i in 1:nbatches
        batch_start = (i-1) * batchsize + 1
        batch_end = min(i * batchsize, length(shuffled))
        mols = shuffled[batch_start:batch_end]
        batch = MOGMOG.pad_and_batch(mols, vocab_dict; random_rigid=true) |> gpu
        loss_val, grad = Flux.withgradient(model) do m
            loss_atom_type, loss_position, loss_climb = losses(m, batch)
            i % 50 == 0 && Flux.ChainRulesCore.@ignore_derivatives println(
                "Batch $i, loss_atom_type = $(round(loss_atom_type, digits=4)), loss_position = $(round(loss_position, digits=4)), loss_climb = $(round(loss_climb, digits=4)), lr = $(round(scheduler.lr, digits=6))")
            loss_atom_type + loss_position + loss_climb
        end
        Flux.update!(opt_state, model, grad[1])
        Flux.adjust!(opt_state, next_rate(scheduler))
        push!(all_losses, loss_val)
        if i % 250 == 0
            samp = MOGMOG.sample(20, cpu(model))
            write("samples/$(epoch)_$(i).xyz.txt", MOGMOG.to_xyz([reverse_vocab_dict[q] for q in samp[1]], samp[2]))
        end
    end
    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss", ylims=(minimum(all_losses) - 1, 10), label = :none)
    savefig("training_loss.pdf")
    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss", label = :none)
    savefig("training_loss_full.pdf")
    let model = model |> cpu
        @save "mogmodel_checkpoint_$(epoch).jld2" model all_losses
    end
end



atom_logits, μ, σ, logw, climb_logits = model(batch.atom_types, batch.positions, batch.climbs, batch.anchors, batch.indexes, batch.displacements)

atom_logits, μ, σ, logw, climb_logits = model(batch.atom_types, batch.positions, batch.climbs, batch.anchors, batch.indexes)

running_median = [median(all_losses[i:i+100]) for i in 1:length(all_losses)-100]
plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss", ylims=(minimum(all_losses) - 0.1, 7.5), label = :none)
plot!(running_median, color = :red, label = :none)
savefig("training_loss.pdf")
plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss", label = :none)
savefig("training_loss_full.pdf")

#

cpumodel = cpu(model)
samp = MOGMOG.sample(20, cpumodel)
println(MOGMOG.to_xyz([reverse_vocab_dict[q] for q in samp[1]], samp[2]))


samp = MOGMOG.sample(20, model)
sqrt(sum((samp[2][:,2] .- samp[2][:,1]).^2))
sqrt.(sum(batch.displacements[:,1,:].^2, dims = 1))
println(MOGMOG.to_xyz([reverse_vocab_dict[q] for q in samp[1]], samp[2]))

sqrt(sum((samp[2][:,3] .- samp[2][:,2]).^2))
sqrt.(sum(batch.displacements[:,2,:].^2, dims = 1))


println(MOGMOG.to_xyz(data[10000].atom_types, data[10000].positions))


i = 1
shuffled = shuffle(data)
batch_start = (i-1) * batchsize + 1
batch_end = min(i * batchsize, length(shuffled))
mols = shuffled[batch_start:batch_end]
batch = MOGMOG.pad_and_batch(mols, vocab_dict; random_rigid=true) |> gpu



ind = 3
batch.climbs[:,ind]
println(MOGMOG.to_xyz([reverse_vocab_dict[q] for q in batch.atom_types[:,ind]], batch.positions[:,:,ind]))


function mogplot!(μ, σ, logw; xrange = -2:0.01:2, kwargs...)
    ys = [MOGMOG.mog_logpdf(x, μ, σ, logw)[1] for x in xrange]
    plot!(xrange, exp.(ys); kwargs...)
end


mols = data[end-100:end-100]
batch = MOGMOG.pad_and_batch(mols, vocab_dict; random_rigid=true)
atom_logits, μ, σ, logw, climb_logits = model(batch.atom_types, batch.positions, batch.climbs)

MOGMOG.logitsample(logw[:,3,:,1])




function mogplot!(μ, σ, logw; xrange = -2:0.01:2, kwargs...)
    ys = [MOGMOG.mog_logpdf(x, μ, σ, logw)[1] for x in xrange]
    plot!(xrange, exp.(ys); kwargs...)
end
    
atom_logits, mμ, mσ, logw, climb_logits = model(batch.atom_types, batch.positions, batch.climbs)
ax = 3
pl = plot(title = ["X", "Y", "Z"][ax])
for com in 1:size(mμ,3)-1
    mogplot!(mμ[:,ax,com,1], mσ[:,ax,com,1], logw[:,ax,com,1], label = "$com")
end
pl

ind = 1
batch.climbs[:,ind]
println(MOGMOG.to_xyz([reverse_vocab_dict[q] for q in batch.atom_types[:,ind]], batch.positions[:,:,ind]))





atom_types = ones(Int, 1, 1)
positions = rand(Float32, 3, size(atom_types)...)
climbs = zeros(Int, 1, 1)
new_atom_types = [atom_types; 1]
new_climbs = [climbs; 0]
from = first.(MOGMOG.climbs_to_pairs(new_climbs[2:end]))[end] #<- sus
new_positions = [positions;; selectdim(positions, 2, from)]
atom_logits1, μ1, σ1, logw1, climb_logits1 = model(new_atom_types, new_positions, new_climbs)
new_positions[1, end, :] .+= MOGMOG.nucleus_mog_sample(μ1[:,1,end,1], σ1[:,1,end,1], logw1[:,1,end,1], p = 0.01, n = 5000)
atom_logits2, μ2, σ2, logw2, climb_logits2 = model(new_atom_types, new_positions, new_climbs)
new_positions[2, end, :] .+= MOGMOG.nucleus_mog_sample(μ2[:,2,end,1], σ2[:,2,end,1], logw2[:,2,end,1], p = 0.01, n = 5000)
atom_logits3, μ3, σ3, logw3, climb_logits3 = model(new_atom_types, new_positions, new_climbs)
new_positions[3, end, :] .+= MOGMOG.nucleus_mog_sample(μ3[:,3,end,1], σ3[:,3,end,1], logw3[:,3,end,1], p = 0.01, n = 5000)

μ2 .- μ1
σ2 .- σ1
logw2 .- logw1

μ3 .- μ2
σ3 .- σ2
logw3 .- logw2

sqrt(sum((new_positions[:,2,1] .- new_positions[:,1,1]).^2))






#pair_features 
#Anchors
#Distances (transformed, I guess?)
#Absolute-coordinate location offset?
#Primary sequence differences? Nah... We have RoPE, and primary seq isn't suuuper important if you have anchors

batched_pairs(operator, a, b) = operator.(reshape(a, 1, :, size(a,2)),reshape(b, :, 1, size(b,2)))

function pair_encode(resinds, chainids)
    chain_diffs = Float32.(batched_pairs(==, chainids, chainids))
    num_diffs = Float32.(batched_pairs(-, resinds, resinds))
    decay_num_diffs = (sign.(num_diffs) ./ (1 .+ abs.(num_diffs) ./ 5)) .* chain_diffs
    return vcat(reshape(decay_num_diffs, 1, size(decay_num_diffs)...), reshape(chain_diffs, 1, size(chain_diffs)...))
end



log.(1.443f0 .* softplus.(.- (pairwise_sqeuclidean(permutedims(batch.positions[:,1:end-1, :], (2,1,3)), batch.positions[:,1:end-1, :]) ./ 5)) )


decay(d) = (sign.(d) ./ (1 .+ abs.(d) ./ 1))
#NOTE: THE ANCHORS SHOULD ACTUALLY BE THE ANCHORS FOR THE NEXT ATOM, BECAUSE THAT IS WHAT THE MODEL NEEDS
#a1 = rearrange(Onion.batched_pairs(==, batch.indexes, batch.anchors), (..) --> (1, ..)) #This one gets the info masked out, so we'll chuck it.
a = rearrange(Onion.batched_pairs(==, batch.anchors, batch.indexes), (..) --> (1, ..))
o1 = rearrange(batch.positions[:,1:end-1, :], (:d, :L, :B) --> (:d, 1, :L, :B)) .- rearrange(batch.positions[:,1:end-1, :], (:d, :L, :B) --> (:d, :L, 1, :B))
o2 = rearrange(batch.positions[:,1:end-1, :], (:d, :L, :B) --> (:d, :L, 1, :B)) .- rearrange(batch.positions[:,1:end-1, :], (:d, :L, :B) --> (:d, 1, :L, :B))
e1 = rearrange(1.443f0 .* softplus.(.- (pairwise_sqeuclidean(permutedims(batch.positions[:,1:end-1, :], (2,1,3)), batch.positions[:,1:end-1, :]) ./ 5)), (..) --> (1, ..))



decay(d) = (sign.(d) ./ (1 .+ abs.(d) ./ 1))
function pair_features(positions, anchors, indexes)
    p = positions[:,1:end-1, :]
    a = rearrange(Onion.batched_pairs(==, anchors, indexes), (..) --> (1, ..))
    o = rearrange(p, (:d, :L, :B) --> (:d, 1, :L, :B)) .- rearrange(p, (:d, :L, :B) --> (:d, :L, 1, :B)) #We don't need the other direction on these, because that is just the sign flip
    d = pairwise_sqeuclidean(permutedims(p, (2,1,3)), p)
    e1 = rearrange(1.443f0 .* softplus.(.- (d)), (..) --> (1, ..))
    e2 = rearrange(1.443f0 .* softplus.(.- (d ./ 5)), (..) --> (1, ..))
    e3 = rearrange(1.443f0 .* softplus.(.- (d ./ 15)), (..) --> (1, ..))
    return vcat(a, o, e1, e2, e3)
end

#pf = pair_features(batch.positions, batch.anchors, batch.indexes);



size(pf)



repeat(pf, (:k, :l1, :l2, :b) --> (:k, (5, :l1), (5, :l2), :b))





