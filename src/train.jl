using Flux, JLD2, Random, Statistics, Onion, RandomFeatureMaps, Onion.Einops

include("loss.jl")
include("MOGhead.jl")
include("MOGfoot.jl")
include("model.jl")
include("utilities.jl")

@load expanduser("~/processed_molecules_smiles.jld2") result

struct Molecule
    atoms::Vector{String}
    positions::Matrix{Float64}
    smiles::String
    climbs::Vector{Int}
end
Base.length(mol::Molecule) = length(mol.atoms)

# === Dictionary and Hyperparameters ===
atom_dict = Dict(name => i for (i, name) in enumerate(["C", "F", "H", "N", "O", "STOP"]))
PAD = atom_dict["STOP"]

embed_dim = 64
n_components = 50
vocab_size = length(atom_dict)
depth = 4
max_len = 32
batchsize = 8
nbatches = 1000
nepochs = 100
max_climb = 10

model = MOGMOGModel(embed_dim, n_components, vocab_size; depth=depth)
opt_state = Flux.setup(AdamW(0.001f0), model)

# === Pad Batch ===
function pad_batch(mols::Vector{Molecule})
    B = length(mols)
    atom_ids      = fill(PAD, max_len, B)
    positions     = zeros(Float32, 3, max_len, B)
    climbs        = fill(0, max_len, B)
    atom_mask     = zeros(Float32, max_len - 1, B)
    coord_mask    = zeros(Float32, 1, 1, max_len - 1, B)

    for (i, mol) in enumerate(mols)
        L = min(length(mol.atoms), max_len - 1)
        for j in 1:L
            atom_ids[j, i]  = atom_dict[mol.atoms[j]]
            positions[:, j, i] = Float32.(mol.positions[j, :])
            atom_mask[j, i] = 1.0
            coord_mask[1, 1, j, i] = 1.0

            # Use climb if available; otherwise default to 0
            climbs[j, i] = j <= length(mol.climbs) ? mol.climbs[j] : 0
        end
        atom_ids[L+1, i] = PAD
        climbs[L+1, i] = 0
    end

    target_atoms = atom_ids[2:end, :]
    target_climbs = climbs[2:end, :]

    return atom_ids, positions, target_atoms, climbs, target_climbs, atom_mask, coord_mask
end



# === Training ===
all_losses = Float32[]
for epoch in 1:nepochs
    println("Epoch $epoch")
    shuffled = shuffle(result)

    for i in 1:nbatches
        mols = shuffled[i:i+batchsize-1]

        atom_ids, positions, target_atoms, climbs, target_climbs, atom_mask, coord_mask = pad_batch(mols)

        # Flatten inputs
        atom_seq = vec(atom_ids)
        target_atom_seq = vec(target_atoms)
        climb_seq = vec(climbs)
        target_climb_seq = vec(target_climbs)
        atom_mask_vec = vec(atom_mask)
        coord_mask_vec = vec(coord_mask)

        # One-hot encodings outside gradient
        atom_onehot = Flux.onehotbatch(atom_seq, 1:vocab_size)
        target_atom_onehot = Flux.onehotbatch(target_atom_seq, 1:vocab_size)
        target_climb_onehot = Flux.onehotbatch(target_climb_seq, 0:max_climb)

        # Compute gradients
        loss_val, back = Flux.withgradient(model) do m
            loss_climb(m, positions, atom_onehot, target_atom_seq, climbs, target_climb_seq, atom_mask_vec, coord_mask_vec)
        end

        Flux.update!(opt_state, model, back)
        push!(all_losses, loss_val)
        println("Batch $i, loss = $(round(loss_val, digits=4))")
    end
end



@save "mogmodel_checkpoint_climb.jld2" model all_losses
