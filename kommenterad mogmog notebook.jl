using Pkg # Du aktiverar projektmiljön (så att rätt versioner av paketen används).
Pkg.activate(".")

using MOGMOG # troligen hantering av molekyler
using DLProteinFormats # laddar proteinstrukturdata
using Flux # maskininlärningsbiblioteket i Julia
using Onion # transformerbibliotek för geometriska data (använder layers m.m.)
using RandomFeatureMaps # för Fourier embeddings, dvs göra om en koordinat till en vektor
using Onion.Einops # för tensoromformningar (t.ex. rearrange)

data = DLProteinFormats.load(PDBSimpleFlat500) # hämtar data från PDBSimpleFlat500

locations = [d.locs for d in data] # hämtar varje atoms position och lägger i en lista 
sequences = [d.AAs for d in data] # hämtar aminosyrasekvenserna och lägger i en lista 

vec(locations[1]) # plattar ut infon från protein 1 och gör det till en vector. 1 innebär alltså endast protein 1 och inte alla. 

function Toy2(dim, depth) # funktionen Toy2 (random namn) tar in argumenten dimension och depth, dvs hur många dimensioner varje vektor har (typ antal tal för att beskriva något) och hur många transformer lager modellen har
    layers = (; # en tuple med alla lager 
        loc_rff = RandomFourierFeatures(3 => 64, 0.1f0), # Här görs varje PUNKT om till en vektor med 64 dimensioner. loc_rff är namnet på lagret, RandomFourierFeatures skapar fourier embedding dvs gör om koordinaten till dimensioner, 3-->64 innebär att varje ATOM genomgår embedding till 64 dimensioner (det där med att varje x y z får 64 siffror kommer senare. Detta är Bens kod). 0.1f0 är en parameter som styr variation (kallas "bandbredd")
        loc_encoder = Dense(64 => dim, bias=false),# Skapar ett lager där dimensionen kan ändras beroende på argumentet. Matematiskt detaljerat. 
        transformers = [Onion.TransformerBlock(dim, 8) for _ in 1:depth], # "Skapa en lista med depth stycken TransformerBlocks". for _ in 1:depth betyder att vi gör detta depth antal gånger och _ används när vi inte bryr oss om indexet i loopen (det används inte i koden). Så varje block: använder self-attention: varje atom tittar på alla andra atomer och beräknar vad som är viktigt (d.v.s. deras relationer). Detta görs i flera lager (depth), så att modellen kan bygga en allt rikare förståelse av interaktioner mellan atomer
        AA_decoder = Dense(dim => 20, bias=false), # producerar en 20 dimensionall vector pga 20 aminosyror 
    )
    return Toy2(layers)
end

function (m::Toy2)(locs)  # När ett objekt m av typen Toy2 anropas som en funktion, med ett argument locs, då ska denna kod köras.
    l = m.layers           # Sparar modellens lager i variabeln l för enklare åtkomst.
    x = l.loc_encoder(l.loc_rff(locs))  # det inuti parantesenm gör om varje (x y z) punkt till fourier embeddings, dvs så att den kan ta emot fler dimensioner ish. Det utanför hämtar dimensionen 
    for layer in l.transformers         # Går igenom alla transformerblock i modellen.
        x = layer(x, 0, nothing)        # Skickar datan genom varje block. Parametrar 0 och nothing = ingen tidsstegsmask eller extra kontext används. Måste dock has med. 
    end
    aa_logits = l.AA_decoder(x)         # Projekterar varje output-vektor till 20 dimensioner (en för varje aminosyra).
    return aa_logits                    # Returnerar logits – ej softmaxade sannolikheter för varje aminosyra.
end

Using NNlib  # Importerar neurala nätverks-funktioner (används under huven i t.ex. Dense)

rff_dim = 32                 # Antal dimensioner för Fourier-embedding av varje koordinatkomponent.
embedding_dim = 64           # Slutlig dimension för transformerinput efter projicering.

# skapar lager 
rff = RandomFourierFeatures(1 => rff_dim, 0.1f0)  # Gör om varje enskild koordinat till en sinus/cosinus-vektor med rff_dim dimensioner.
pre_transformer_proj = Dense(rff_dim => embedding_dim, bias=false)  # gör om till transformer-dimension.
transformer_blocks = [DART(TransformerBlock(64, 2)) for i in 1:10]  # Skapar 10 transformerblock. Andra siffran är antal heads, dvs hur många delar embeddings delas upp i och kollas på för att identifierna mönster. Dart formar om datan. 

# transformerar om datan och passerar genom lagrena
input_coordinates = rearrange(locations[1], (:K, 1, :L) --> (:K, :L))  # Tar bort en onödig dimension från koordinaterna (från K×1×3 → K×3).
rotated_coordinates = transform_molecule(input_coordinates)  # Din egen funktion! Roterar/centrerar molekylen så modellen blir invariant för orientering.
coordinate_tokens = rearrange(rotated_coordinates, (:K, :L) --> (1, :K, :L))  # gör datan till en kub där ena sidan bara är 1. Den sidan kommer sedan göras till 32 av rff som innehåller våra embeddings
clock_tokens = rff(coordinate_tokens)  # Här går vi från 1 till 32 dimensionerna på kuben 
embeddings = pre_transformer_proj(clock_tokens)  # Projekterar dessa till rätt dimension för transformerlagren (t.ex. 64-dimensionella).

for block in transformer_blocks
    embeddings = block(embeddings)  # Skickar genom embeddings genom varje block 
end

embeddings  # Visar slutgiltiga embeddings för alla atomer i molekylen, efter att ha gått genom transformern. Det är slutgiltiga kuben! Den går vidare till nästa steg. 

Onion.causal_mask(rand(5,5))  # Skapar en "causal mask" (för sekvensmodeller): döljer framtida tokens så modellen bara ser bakåt.

