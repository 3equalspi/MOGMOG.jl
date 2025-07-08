using Einops # For reshaping 

struct MOGfoot # strukturera footen 
    current_coord_embed # Embeddar nuvarande atomers (x, y, z)-koordinater
    atom_embed # Embedding för atomtyper (t.ex. kol, väte, kväve). Både nuvarande och näst atom.
    position_embed # Nuvarande atoms position, index 1,2,3...
    next_x_embed # nästa atoms x koordinat 
    next_y_embed # nästa atoms y koordinat 
end

function MOGfoot(embed_dim::Int, vocab_size::Int) # En funktion som skapar olika lager för foten: 
    MOGfoot(
        Chain(RandomFourierFeatures(3 => embed_dim), Dense(embed_dim => embed_dim)), # Ett lager som gör nuvarande atomens (x,yz,) koordinater (representeras av 3-->) till D dimensioner genom icke linjär progression, dvs cosinus och sinus för att få mer exakt info. Sedan kommer dense att göra datan mer läsbar för transformern. Dvs en vektor. 
        Embedding(vocab_size => embed_dim), # Skapar ett lager som embeddar atomens atomtyp och gör den till rätt dimension. 
        Chain(RandomFourierFeatures(1 => embed_dim), Dense(embed_dim => embed_dim)), # Ett lager som returnerar vektorer som beskriver atomens position
        Chain(RandomFourierFeatures(1 => embed_dim), Dense(embed_dim => embed_dim)), # Ett lager som returnerar en vektor som beskriver nästa x genom skillnaden i x-position mellan nuvarande och nästa atom.
        Chain(RandomFourierFeatures(1 => embed_dim), Dense(embed_dim => embed_dim)), # Ett lager som returnerar en vektor som beskriver nästa y genom skillnaden i y-position mellan nuvarande och nästa atom.
    )
end

# atom types: L x B
# [1, 5, 3, 7, 8]
# coordinates: 3 x L x B
# 3×5 Matrix{Float64}:
#  0.163998   0.324916   0.866443  0.0771369  0.0660904
#  0.306506   0.0206321  0.417595  0.76132    0.203126
#  0.0464874  0.290774   0.102366  0.206444   0.338374
function (foot::MOGfoot)(atom_types::AbstractArray{Int}, coordinates::AbstractArray{<:AbstractFloat}) # Här används lagrena. En funktion som totalt sett konverterar atomtyper (L×B) och koordinater (3×L×B) till en tensor med shape (D×4×L×B), dvs varje atom L representeras med D embeddingar som beskriver 4: atom, next_atom, next_x, next_y (B är batch men inte lika viktigt)
    L = length(atom_types) # längden dvs antal atomer 
    # D x L x B
    atom_embedding = foot.atom_embed(atom_types[1:L-1,:]) .+
        foot.current_coord_embed(coordinates[:,1:L-1,:]) .+
        foot.position_embed(reshape(1:L-1, 1, :)) # denna rad embeddar information om varje atom vilket representeras av (1:L-1) men inte L eftersom det blir nästa atom. Vi embeddar alltså atomtyp, koordinater och index i samma för varje atom. 
    # D x L x B --> D x 4 x L x B
    with_next_atom = atom_embedding + foot.atom_embed(atom_types[2:L]) # tar förra atomen och lägger till nästa atoms atomtyp 
    with_next_x = with_next_atom + foot.next_x(coordinates[2:L]) # tar förra och lägger till nästa atoms x
    with_next_y = with_next_x + foot.next_y(coordinates[2:L]) # tar förra och lägger till nästa atoms y. Nu har varje atoms embedding info om: Den själv, Nästa atom, Hur den rör sig i x och y
    concatenated = vcat(atom_embedding, with_next_atom, with_next_x, with_next_y) # Klistrar ihop 4 versionerna av varje atom. 4D x L x B
    return rearrange(concatenated, einops"(d k) l b -> d k l b", k=4) # Omformar till D×4×(L−1)×B dvs dimensioner, de 4 atom_embedding, with_next_atom, with_next_x, with_next_y, antal atomer, batch. Enklare att tolka! 
end
