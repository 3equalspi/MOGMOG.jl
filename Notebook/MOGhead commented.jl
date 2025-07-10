struct MoGAxisHead # En struct/typ som innehåller 3 lager. Själva strukturen på vår gruppering. 
    linear_μ::Dense         # A layer that predict means
    linear_σ::Dense      # A layer that predict σ (positive after softplus)
    linear_logw::Dense      # A layer that predict unnormalized log-weights
end

function MoGAxisHead(embed_dim::Int, n_components::Int) # Funktion som tar in antal embedding dimensioner och antal normalfördelningar. Använder själva strukturen ovan och för att veta hur den ska se ut. 
    return MoGAxisHead( # Här får vi ut våra 3 kuber (arrays) med normalfördelningar, xyz koordinater och atomer. Detta fås alltså av vår slutgiltiga kub från förra delen. Då var däremot en axel våra output embeddings och inte normalfördelningar. 
        Dense(embed_dim, n_components),# Returnerar ett lager som förutspår μ_k med ett visst antal embedding dimensions och normalfördelningar. 
        Dense(embed_dim, n_components, softplus), # Returnerar ett lager som förutspår σ_k med ett visst antal embedding dimensions och normalfördelningar. Softplus används för att se till att den plir positiv. 
        Dense(embed_dim, n_components),# Returnerar ett lager som förutspår log w_k med ett visst antal embedding dimensions och normalfördelningar. 
    )
end

function (head::MoGAxisHead)(axis_embeddings::AbstractMatrix) # Detta säger att det som anropas är ett objekt av typen MoGAxisHead. Du ger det lokala namnet head. Andra parantesten innebär att när MoGAxisHead kallas på så körs denna funktion och axis_embeddings skickas in som en matris av (embed_dim, L)

    μ = head.linear_μ(axis_embeddings) # head är ett objekt med 3 lager. Här väljer vi att skicka in embeddings i medelvärde lagret
    σ = head.linear_σ(axis_embeddings) # Här väljer vi att skicka in embeddings i varians lagret
    logw = head.linear_logw(axis_embeddings) # Här väljer vi att skicka in embeddings i log w lagret
    logw = logw .- logsumexp(logw; dims=1) # Denna rad ska normalisera log w. Kanske inte så viktig så fastna ej på. 

    return μ, σ, logw
end

function AtomTypeHead(embed_dim::Int, vocab_size::Int) # ett lager som returnerar en vektor som förutspår sannolikheter (logits) för vilken typ av atom varje embedding representerar.
    return AtomTypeHead(Dense(embed_dim, vocab_size; bias = false))  #(V, L)
end

function (head::AtomTypeHead)(embeddings::AbstractMatrix) # Den kör helt enkelt embeddingarna genom Dense-lagret och vi frå ut en matris av (vocab_size, L) dvs. För varje atom 1,2,3... (en kolumn), får du ett sannolikhetsvärde att den atomen är just en atomtyp (rad). vocab_size är alltså C,H,O i rader. 
    return head.linear_logits(embeddings)
end


# Först: en kub där ena axeln är output embeddings, atomerna och xyz. Vi har 64 siffror för varje koordinat som tillsammans beskriver: spatial kontext, närliggande atomer, typ av aminosyra, ev. sekvensinfo eller kemisk miljö. De är inte sannolikheter. De är inte koordinater direkt. De är "features" – tänk som en samling beskrivande drag som modellen kan använda vidare.
# Sedan: Nu tar vi varje embedding (64-dim) och skickar den genom 3 olika Dense-lager:
# Vi får 3 nya kuber. En för normalfördelning, varians och w. 
# För varje embedding har alltså ett möjligt medelvärde, varians och w räknats ut. 



