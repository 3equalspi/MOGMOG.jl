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

function (head::MoGAxisHead)(axis_embeddings::AbstractMatrix)
    # axis_embeddings: (embed_dim, L)

    μ = head.linear_μ(axis_embeddings)
    σ = head.linear_σ(axis_embeddings) 
    logw = head.linear_logw(axis_embeddings)     
    logw = logw .- logsumexp(logw; dims=1)       # Normalize log w over K

    return μ, σ, logw
end

# Först: en kub där ena axeln är output embeddings, atomerna och xyz. Vi har 64 siffror för varje koordinat som tillsammans beskriver: spatial kontext, närliggande atomer, typ av aminosyra, ev. sekvensinfo eller kemisk miljö. De är inte sannolikheter. De är inte koordinater direkt. De är "features" – tänk som en samling beskrivande drag som modellen kan använda vidare.
# Sedan: Nu tar vi varje embedding (64-dim) och skickar den genom 3 olika Dense-lager:
# Vi ser alltså för varje atom, 3 st xyz koordinater som beskrivs med massa siffror och vart de ligger .....? 

