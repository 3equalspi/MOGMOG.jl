using NNlib  # För att använda logsumexp

function logpdf_MOG(x::Float64,
                    mus::Vector{Float64},
                    sigmas::Vector{Float64},
                    log_weights::Vector{Float64})

    L = @. log_weights - log(sigmas) - 0.5 * log(2π) - (x - mus)^2 / (2 * sigmas^2) # Vektoriserad (dvs användning av . (elementvis) för att minska kod och göra det smidigare) beräkning av log-pdf för varje Gaussisk komponent. Detta ersätter for loopen. 
    # att använda @. är samma som L = log_weights .- log.(sigmas) .- 0.5 .* log(2π) .- (x .- mus).^2 ./ (2 .* sigmas.^2)

    return logsumexp(L) # Summera sannolikheterna i log-rummet (log-sum-exp trick)
end

# DVS skillnaderna mellan versionerna: 
# 1) For loopen ersätts med punktnotation som sumeras till @. 
# 2) Vi importerar ett bibliotek för logsumexp 

# \mu + TAB → μ
# \sigma + TAB → σ
# \pi + TAB → π
# för att få \: option shift 7 

# för att importera paket så som NNlib, i terminalen: 
# "julia"
# "using Pkg"
# "Pkg.add("NNlib")
# klart

# sätt att runna Julia: 
# "julia"
# 2+2 
# 4 

# play knappen 

# julia MOGMOG.jl om filen är döpt till jl 