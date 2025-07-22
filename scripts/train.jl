using MOGMOG

using Flux, CUDA
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
    climbs = MOGMOG.smiles_to_climbs(mol.smiles)
    edges = MOGMOG.climbs_to_pairs(climbs)
    return (; atom_types, positions, climbs, edges)
end
data = prepare.(molecules)

unique_atoms = unique(vcat((mol.atom_types for mol in data)...))
vocabulary = sort([unique_atoms; "H"; "STOP"])
vocab_dict = Dict(name => i for (i, name) in enumerate(vocabulary))
reverse_vocab_dict = Dict(i => name for (i, name) in enumerate(vocabulary))

# Hyperparameters
embed_dim = 192
mixture_components = 32
vocab_size = length(vocab_dict)
depth = 12
heads = 8
batchsize = 32
nbatches = 1000
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
        batch = pad_and_batch(mols, vocab_dict; random_rigid=true) |> gpu

        loss_val, (grad,) = Flux.withgradient(model) do m
            loss_atom_type, loss_position, loss_climb = losses(m, batch)
            loss_climb *= 2
            i % 50 == 0 && Flux.ChainRulesCore.@ignore_derivatives println(
                "Batch $i, loss_atom_type = $(round(loss_atom_type, digits=4)), loss_position = $(round(loss_position, digits=4)), loss_climb = $(round(loss_climb, digits=4)), lr = $(round(scheduler.lr, digits=6))")
            loss_atom_type + loss_position + loss_climb
        end

        Flux.adjust!(opt_state, next_rate(scheduler))
        Flux.update!(opt_state, model, grad)

        push!(all_losses, loss_val)
    end

    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss", ylims=(minimum(all_losses) - 1, 10))
    savefig("training_loss.pdf")
    
    let model = model |> cpu
        @save "experiments/climb/mogmodel_checkpoint_$(epoch).jld2" model all_losses
    end
end
