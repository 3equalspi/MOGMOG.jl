# Funktion som beräknar log-sannolikheten för en blandning av Gaussiska fördelningar (MoG)
function logpdf_MOG(x::AbstractArray{<:AbstractFloat}, 
    μ::AbstractArray{<:AbstractFloat}, 
    σ::AbstractArray{<:AbstractFloat},
    logw::AbstractArray{<:AbstractFloat})

# Beräkna log-sannolikheten enligt mixad Gaussisk sannolikhetsformel
return logsumexp((@. logw - log(σ) - 0.5 * log(2π) - (x - μ)^2 / (2 * σ^2)), dims=1)
end

export loss  # Gör loss-funktionen tillgänglig utanför modulen

# Förlustfunktion som kombinerar positionsförlust (regression) och atomtypförlust (klassificering)
function loss(model, atom_ids, pos, atom_mask, coord_mask)
target_atoms = atom_ids[2:end, :]  # Sanna (nästa) atomtyper att förutsäga

μ, σ, logw, logits = model(pos, atom_ids)  # 

# Beräkna rörelser (displacement) mellan efterföljande atomer: Δx = x₂ − x₁ osv.
disp = pos[:, 2:end, :] .- pos[:, 1:end-1, :]          
disp = reshape(disp, 1, size(disp)...)  # Ändra form. Form: (1, 3, L-1, B)

# Log-sannolikhet för dessa displacement enligt modellens Gauss-komponenter
logp_xyz = logpdf_MOG(disp, μ, σ, logw)
@show size(logp_xyz)  # Debug: visa tensorstorlek

# Förlust för positioner: maska bort padding och ta negativ medelvärde
loss_xyz = -sum(logp_xyz .* reshape(coord_mask, 1, 1, size(coord_mask)...)) / sum(coord_mask)

# Gör one-hot representation av sanna nästa atomer
atom_onehot = Flux.onehotbatch(target_atoms, 1:size(logits, 1))

# Anpassa masken så att den matchar dimensionskraven
atom_mask = reshape(atom_mask, 1, size(atom_mask)...)

# Beräkna medelvärde där masken är 1 
masked_mean(p) = sum(p .* atom_mask) / sum(atom_mask)

# Klassificeringsförlust: jämför logits mot one-hot med maskerad cross-entropy
loss_type = Flux.logitcrossentropy(dropdims(logits, dims=2), atom_onehot; agg=masked_mean)

return loss_xyz + loss_type  # Total förlust = position + atomtyp
end
