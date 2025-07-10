using MOGMOG

using Flux
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
    return Molecule(atom_types, positions)
end

data = [get_molecule(dataset, i) for i in 1:length(dataset)]

unique_atoms = sort(unique(vcat((mol.atoms for mol in data)...)))
vocabulary = [unique_atoms; "STOP"]
vocab_dict = Dict(name => i for (i, name) in enumerate(vocabulary))

# Hyperparameters
embed_dim = 64
mixture_components = 5
vocab_size = length(vocab_dict)
depth = 4
heads = 4
batchsize = 64
nbatches = 10000
nepochs = 10

ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

# Initialize model and optimizer
model = MOGMOGModel(; embed_dim, mixture_components, vocab_size, depth, heads) |> gpu

scheduler = burnin_learning_schedule(0.00005f0, 0.001f0, 1.01f0, 0.9998f0)
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
        batch = pad_and_batch(mols, vocab_dict; random_rigid=true)

        loss_val, (grad,) = Flux.withgradient(model) do m
            loss(m, batch...)
        end

        Flux.adjust!(opt_state, next_rate(scheduler))
        Flux.update!(opt_state, model, grad)

        push!(all_losses, loss_val)
        println("Batch $i, loss = $(round(loss_val, digits=4)), lr = $(round(scheduler.lr, digits=6))")
    end

    plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss")
    savefig("training_loss.pdf")
    
    let model = model |> cpu
        @save "mogmodel_checkpoint_$(epoch).jld2" model all_losses
    end
end
