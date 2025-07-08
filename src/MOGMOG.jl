module MOGMOG

using LinearAlgebra
using Flux
using NNlib
using Random
using Onion
using RandomFeatureMaps

include("pdf.jl")
export logpdf_MOG

include("MOGhead.jl")
export MoGAxisHead

include("MOGfoot.jl")

include("model.jl")
export MOGMOGModel

include("utilities.jl")
export transform_molecule
export loss_fn

end
