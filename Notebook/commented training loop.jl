###TRAINING loop

using Flux, JLD2, StatsBase, Random, Plots
losses_over_time = Float32[]

# Load processed molecules
@load "processed_molecules.jld2" result  # Loads `result::Vector{Molecule}`

# Atom dictionary: atom => integer index
atom_dict = Dict("C"=>1, "O"=>2, "N"=>3, "H"=>4, "F"=>5, "STOP"=>6)  # Customize this

# Model
embed_dim = 64
n_components = 5
vocab_size = length(atom_dict)
model = MOGMOGModel(embed_dim, n_components, vocab_size)

# Optimizer
opt_state = Flux.setup(AdamW(0.001), model)

# Hyperparameters
nepochs = 10 # How many times we look at the data
nbatches = 1000 # Number of batches per epoch
batchsize = 16 # Size of each batch. Meaning we look at 16 000 molecules 10 times. 

# Training loop
for epoch in 1:nepochs # Looks at the data 10 times 
    @info "Epoch $epoch"
    shuffled = shuffle(result) # Shuffle the molecules so that the model doesnt remember the order 
    total_loss = 0.0

    for i in 1:nbatches # For every batch 
        mols = shuffled[(i-1)*batchsize+1:min(i*batchsize, end)] # Take 16 molecules from the shuffled list 
        loss_batch = 0.0

        grads = Flux.gradient(model) do m # Calculate all derivatives (gradients) of the model 
            losses = Float32[]
            for mol in mols # Calculate loss for every molecule 
                try
                    l = loss_fn(m, mol, atom_dict)
                    push!(losses, l)
                catch e # If the molecule is broken psuh an error 
                    @warn "Failed on molecule: $e"
                end
            end
            loss_batch = mean(losses) # Take the mean of the loss
            push!(losses_over_time, loss_batch) # Save it in losses_over_time which is used for the plotting
            return loss_batch
        end

        Flux.update!(opt_state, model, grads[1])#  It goes through every parameter in your model (weights, biases, etc.) and updates them using the gradient and your optimizer settings. Gradients tell you how much each weight should be adjusted, model holds the weights, and opt state is the optimizer you are using. 
        total_loss += loss_batch # lägg till lossen 

        if i % 50 == 0 # print for every 50th batch 
            println("Batch $i, loss = $(round(loss_batch, digits=4))")
        end
    end

    println("Epoch $epoch done. Avg loss = $(round(total_loss / nbatches, digits=4))")
end

plot(losses_over_time, xlabel="Batch", ylabel="Loss", label="Training Loss", title="Final Loss Curve")
savefig("final_loss_curve.png") # plot 
