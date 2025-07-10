using GLMakie, JLD2

#struct Molecule
#    atoms::Vector{String}
#    positions::Matrix{Float64}
#end

# === Ladda molekyl ===
#@load "/Users/alicestenbeck/Downloads/processed_molecules.jld2" result
#mol = result[1]
mol_length = 10
ghost_number = 50

mol_atoms = rand(string.(collect("CHON")), mol_length) # e.g. ["C", "H", "O"] # length N
mol_positions = randn(mol_length, 3) * 1.5             # e.g. N x 3

ghost_atoms = rand(string.(collect("CHON")), ghost_number) # ["C", "C", "C", "C", "O"] # length M
ghost_positions = randn(ghost_number, 3) * 3               # e.g. M x 3

# === Färger och radier ===
atom_colors = Dict("H" => :white, "C" => :gray, "O" => :red, "N" => :blue, "F" => :lightblue)

# === Konvertera till punkter ===
positions = [Point3f(mol.positions[i, :]...) for i in 1:size(mol.positions, 1)]
colors = [atom_colors[a] for a in mol.atoms]

# === Rita plotten ===
fig = Figure(resolution=(600, 600))
ax = Axis3(fig[1, 1], aspect=:data)
scatter!(ax, positions, color=colors, markersize=20, marker=:circle)
fig
