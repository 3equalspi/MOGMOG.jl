function logpdf_MOG(x::AbstractArray{Float64}, 
                        μ::AbstractArray{Float64}, 
                        σ::AbstractArray{Float64},
                        logw::AbstractArray{Float64})
    return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end
