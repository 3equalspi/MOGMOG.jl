using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../")

#Pkg.add(["Flux", "LearningSchedules", "Random", "JLD2", "Serialization", "PeriodicTable", "Plots", "CannotWaitForTheseOptimisers"])

using MOGMOG

using Flux
using LearningSchedules
using Random
using JLD2
using Serialization
using PeriodicTable
using Plots
using CannotWaitForTheseOptimisers

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

# Hyperparameters
embed_dim = 128
mixture_components = 32
vocab_size = length(vocab_dict)
depth = 6
heads = 8
batchsize = 8
nbatches = 10000
nepochs = 100

ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# Initialize model and optimizer
model = MOGMOGModel(; embed_dim, mixture_components, vocab_size, depth, heads) |> gpu;

scheduler = burnin_learning_schedule(0.00003f0, 0.001f0, 1.01f0, 0.99995f0)
opt_state = Flux.setup(Muon(opt=AdamW(scheduler.lr)), model);

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
        loss_val, (grad,) = Flux.withgradient(model) do m
            loss_atom_type, loss_position, loss_climb = losses(m, batch)
            i % 50 == 0 && Flux.ChainRulesCore.@ignore_derivatives println(
                "Batch $i, loss_atom_type = $(round(loss_atom_type, digits=4)), loss_position = $(round(loss_position, digits=4)), loss_climb = $(round(loss_climb, digits=4)), lr = $(round(scheduler.lr, digits=6))")
            loss_atom_type + loss_position + loss_climb
        end
        Flux.update!(opt_state, model, grad)
        Flux.adjust!(opt_state, next_rate(scheduler))
        push!(all_losses, loss_val)
        if i % 100 == 0
            samp = MOGMOG.sample(20, model)
            write("samples/$(epoch)_$(i).xyz.txt", MOGMOG.to_xyz([reverse_vocab_dict[q] for q in samp[1]], samp[2]))
        end
    end
    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss", ylims=(minimum(all_losses) - 1, 10))
    savefig("training_loss.pdf")
    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss")
    savefig("training_loss_full.pdf")
    let model = model |> cpu
        @save "mogmodel_checkpoint_$(epoch).jld2" model all_losses
    end
end





samp = MOGMOG.sample(20, model)
println(MOGMOG.to_xyz([reverse_vocab_dict[q] for q in samp[1]], samp[2]))



println(MOGMOG.to_xyz(data[10000].atom_types, data[10000].positions))


batch_start = (i-1) * batchsize + 1
        batch_end = min(i * batchsize, length(shuffled))
        mols = shuffled[batch_start:batch_end]
        batch = MOGMOG.pad_and_batch(mols, vocab_dict; random_rigid=true) |> gpu

ind = 3
batch.climbs[:,ind]
println(MOGMOG.to_xyz([reverse_vocab_dict[q] for q in batch.atom_types[:,ind]], batch.positions[:,:,ind]))