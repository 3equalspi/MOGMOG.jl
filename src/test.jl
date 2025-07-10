using Flux, JLD2, Random, Statistics, Onion, RandomFeatureMaps, Onion.Einops
using Pkg
Pkg.add("Plots")

include("loss.jl")
include("MOGhead.jl")
include("MOGfoot.jl")
include("model.jl")
include("utilities.jl")

export logpdf_MOG, MoGAxisHead, MOGMOGModel, transform_molecule, loss

# Load molecules
@load expanduser("~/processed_molecules.jld2") result

struct Molecule
    atoms::Vector{String}
    positions::Matrix{Float64}
end

Base.length(mol::Molecule) = length(mol.atoms)


# Atom dictionary
atom_dict = Dict(name => i for (i, name) in enumerate(["C", "F", "H", "N", "O", "STOP"]))
PAD = atom_dict["STOP"]

# Hyperparameters
embed_dim = 64
n_components = 5
vocab_size = length(atom_dict)
depth = 4
max_len = 32
batchsize = 4
nbatches = 1000
nepochs = 100

# Initialize model and optimizer
model = MOGMOGModel(embed_dim, n_components, vocab_size, depth=depth)
opt_state = Flux.setup(AdamW(0.001f0), model)

# Padding function
function pad_batch(mols::Vector{Molecule})
    B = length(mols)
    atom_ids = fill(PAD, max_len, B)
    positions = zeros(Float32, 3, max_len, B)
    atom_mask = zeros(Float32, max_len - 1, B)
    coord_mask = zeros(Float32, max_len - 1, B)

    for (i, mol) in enumerate(mols)
        L = min(length(mol.atoms), max_len - 1)
        for j in 1:L
            atom_ids[j, i] = atom_dict[mol.atoms[j]]
            positions[:, j, i] = Float32.(mol.positions[j, :])
            atom_mask[j, i] = 1.0
            coord_mask[j, i] = 1.0
        end
        atom_ids[L+1, i] = PAD
    end
    return atom_ids, positions, atom_mask, coord_mask
end

# Training loop
all_losses = Float32[]
for epoch in 1:nepochs
    println("Epoch $epoch")
    shuffled = shuffle(result)

    for i in 1:nbatches
        mols = shuffled[i:i+batchsize-1]
        atom_ids, pos, atom_mask, coord_mask = pad_batch(mols)

        loss_val, (grad,) = Flux.withgradient(model) do m
            loss(m, atom_ids, pos, atom_mask, coord_mask)
        end

        Flux.update!(opt_state, model, grad)
        push!(all_losses, loss_val)

        println("Batch $i, loss = $(round(loss_val, digits=4))")
    end
end

# Save loss plot
using Plots
plot(all_losses, title="Training Loss", xlabel="Batch", ylabel="Loss")
savefig("training_loss.pdf")

# Save checkpoint
@save "mogmodel_checkpoint.jld2" model all_losses
