module MOGMOG

using LinearAlgebra
using Flux
using NNlib
using Random
using Onion

include("pdf.jl")
export logpdf_MOG

include("MOGhead.jl")
export MoGAxisHead

include("utilities.jl")
export transform_molecule
export loss_fn
end
