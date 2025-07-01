function logpdf_MOG(x::Float64,                        # Compute log-probability density of x under a mixture of Gaussians
    mus::Vector{Float64},                             # Mus = my = means0 medelvärde of each Gaussian component. Makes a vector. 
    sigmas::Vector{Float64},                          # Standard deviations of each Gaussian component. Makes a vector. 
    log_weights::Vector{Float64})                     # Logarithm of the mixture weights for each component. Makes a vector. 
    
    K = length(mus)                                  # Number of Gaussian components in the mixture
    L = Vector{Float64}(undef, K)                    # Pre-allocate vector L to store log-probabilities per component
    
    for k in 1:K # här får vi alltså ett Lk för varje gaussiskt element genom for loopen. Detta är det som skippas i andra versionen genom punkterna. 
        μ = mus[k]                                   # Mean of the k-th Gaussian
        σ = sigmas[k]                                # Std dev of the k-th Gaussian
        
        L[k] = log_weights[k] - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)
    end
    
    M = maximum(L)                                   # Find the maximum log-probability for numerical stability
    
    # Return the log of the weighted sum of probabilities using the log-sum-exp trick:
    # log(sum_k w_k * Normal(x | μ_k, σ_k)) = M + log(sum(exp(L - M)))
    return M + log(sum(exp.(L .- M)))
end