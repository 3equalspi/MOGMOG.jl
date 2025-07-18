using MOGMOG

using Flux, CUDA
using LearningSchedules
using Random
using JLD2
using MLDatasets
using PeriodicTable
using Plots

Random.seed!(0)

# Load data
dataset = TUDataset("QM9")

function get_molecule(data, i)
    features = data[i].graphs.node_data.features
    atom_types = [elements[Int(i)].symbol for i in features[6,:]]
    positions = features[14:16,:]
    i = (:)#findall(!=("H"), atom_types)
    return Molecule(atom_types[i], positions[:,i])
end
data = [get_molecule(dataset, i) for i in 1:length(dataset)]

unique_atoms = unique(vcat((mol.atoms for mol in data)...))
vocabulary = sort([unique_atoms; "H"; "STOP"])
vocab_dict = Dict(name => i for (i, name) in enumerate(vocabulary))

# Hyperparameters
embed_dim = 64
mixture_components = 4
vocab_size = length(vocab_dict)
depth = 4
heads = 8
batchsize = 64
nbatches = 1000
nepochs = 50

ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# Initialize model and optimizer
model = MOGMOGModel(; embed_dim, mixture_components, vocab_size, depth, heads) |> gpu

scheduler = burnin_learning_schedule(0.00005f0, 0.001f0, 1.01f0, 0.9999f0)
opt_state = Flux.setup(AdamW(scheduler.lr), model)

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
            loss_atom_type, loss_position = losses(m, batch...)
            i % 50 == 0 && Flux.ChainRulesCore.@ignore_derivatives println(
                "Batch $i, loss_atom_type = $(round(loss_atom_type, digits=4)), loss_position = $(round(loss_position, digits=4)), lr = $(round(scheduler.lr, digits=6))")
            loss_atom_type + loss_position
        end

        Flux.adjust!(opt_state, next_rate(scheduler))
        Flux.update!(opt_state, model, grad)

        push!(all_losses, loss_val)
    end

    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss")
    savefig("training_loss.pdf")
    
    let model = model |> cpu
        @save "mogmodel_checkpoint_$(epoch).jld2" model all_losses
    end
end
