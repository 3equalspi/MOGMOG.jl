nepochs = 1                 # Antal epoker (hur många gånger vi tränar över hela datan)
nbatches = 100              # Antal batcher per epok (vi delar upp datan i 100 delar)
batchsize = 1               # Hur många molekyler vi tränar på samtidigt (1 = en åt gången)
max_len = 32                # Maxlängd för padding (alla molekyler paddas till längd 32)
all_losses = []             # Här sparas alla loss-värden över tid (för plot eller analys)

# === Träningsloop ===
for epoch in 1:nepochs
    @info "Epoch $epoch"
    shuffled = shuffle(result)  # Blanda molekylerna slumpmässigt varje epok för att undvika att modellen lär sig ordningen
    
    epoch_losses = []           # Här sparas alla loss-värden för just denna epok

    for i in 1:nbatches
        mols = shuffled[i:i+batchsize-1]  # Välj batchsize stycken molekyler (just nu bara 1 molekyl per batch)

        # Pad molekylerna så att de får samma längd → returnerar:
        # atom_ids:    index av atomer (t.ex. C = 1, O = 2, ...)
        # positions:   3D-koordinater för varje atom
        # atom_mask:   vilka atompositioner som är riktiga (inte PAD)
        # coord_mask:  vilka koordinater som är riktiga (inte PAD)
        atom_ids, positions, atom_mask, coord_mask = pad_batch(mols, atom_dict, max_len)

        # Beräkna förlust och gradienter (automatisk derivering)
        # grad_model innehåller gradienten för varje parameter i modellen
        loss_batch, (grad_model,) = Flux.withgradient(model) do m
            loss(m, atom_ids, positions, atom_mask, coord_mask)
        end

        # Uppdatera modellens parametrar med gradienterna
        #   model       = håller vikterna
        #   grad_model  = gradienter (hur mycket varje vikt ska ändras)
        #   opt_state   = din optimerare (t.ex. AdamW) som bestämmer *hur* vikterna uppdateras
        Flux.update!(opt_state, model, grad_model)

        # Spara förlusten för denna batch
        push!(all_losses, loss_batch)
        push!(epoch_losses, loss_batch)

        # Print för att hålla koll på hur det går under träningen
        println("Batch $i, loss = $(round(loss_batch, digits=4))")
    end

    # När en epok är klar, skriv ut medel-förlusten
    println("Epoch $epoch done. Avg loss = $(round(mean(epoch_losses), digits=4))")
end


atom_dict = Dict("C"=>1, "F"=>2, "H"=>3, "N"=>4, "O"=>5, "STOP"=>6)
PAD = atom_dict["STOP"]
vocab_size = length(atom_dict)

# Pad function to make the molecules the same lngth and use PAD if there are no more atoms. 
function pad_batch(mols::Vector{Molecule}, atom_dict::Dict{String,Int})
    max_len = maximum(length.(mols)) + 1  # Find the maximum length of molecules
    B = length(mols) 
    atom_types = fill(PAD, max_len, B) # en matris max_len *B där allt blir PAD till att börja med och det ändras sedan. Modellen behöver veta vilken typ av atom det är på varje plats i molekylen för att kunna förutspå nästa atom.
    coordinates = zeros(Float32, 3, max_len, B) .+ 10 # en matris av 3*max_len* B som fylls med 0 enligt zeros med sedan 10 som bara ett värde. Modellen måste veta var atomerna ligger i rymden för att kunna lära sig struktur.
    atom_type_mask = zeros(Float32, max_len - 1, B) # en matris av max_len-1 * B som fylls med 0 till att börja med. Behövs för att bara räkna på 1or 
    coordinate_mask = zeros(Float32, max_len - 1, B)# samma här 

    for (i, mol) in enumerate(mols) # loopar över varje molekyl och hämtar indexet på molekylen men även mols dvs en lista på atomer och en matris med koordinater
        L = length(mol) # antal atomer i molekylen INNAN PADDING 
        for j in 1:L # loopar över varje atom i molekylen 
            atom_types[j, i] = get(atom_dict, mol.atoms[j], PAD) # atom_types är en 2D matris av i rader (index på atomen) och j kolumner (index på vilken molekyl). Den andra delen hämtar vad atomtypen har för tal tex c=1 och h=3 och om det inte finns så blir det PAD
        end
        coordinates[:, 1:L, i] = mol.positions[1:L, :]' # coordinates i 3*max_len*B sparar alla molekylers koordinater 
        coordinate_mask[1:L-1, i] .= 1.0 # Gå till kolumn i i coordinate_mask (en molekyl i batchen). Sätt raderna 1 till L-1 (de verkliga koordinaterna) till 1.0. Resten av kolumnen förblir 0.0 (och betyder padding). (Eftersom L var längen av molekylen INNAN padding). L-1 eftersom det bara finns så många displacements. 
        atom_type_mask[1:L, i] .= 1.0 # Gå till kolumn i i atom_type_mask (en molekyl i batchen). Sätt raderna 1 till L (de verkliga atomerna) till 1.0. Resten av kolumnen förblir 0.0 (och betyder padding). (Eftersom L var längen av molekylen INNAN padding)
    end
    return atom_types, coordinates, atom_type_mask, coordinate_mask
end
# Model
embed_dim = 64
n_components = 5
vocab_size = length(atom_dict)
model = MOGMOGModel(embed_dim, n_components, vocab_size, depth=4)

# Optimizer
opt_state = Flux.setup(AdamW(0.001f0), model)
