using NNlib

function logpdf_MOG(x::Float64, 
                        μ::Vector{Float64}, 
                        σ::Vector{Float64}, 
                        logw::Vector{Float64})
    L = @. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)
    return logsumexp(L)
end
