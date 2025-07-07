using Pkg
Pkg.activate("visuals", shared=true)

using GLMakie
using LinearAlgebra

# --------------------------
# Atomtyper (Etanol: C2H5OH)
atom_types = [
    "C",  # C1
    "C",  # C2
    "O",  # O
    "H",  # H1
    "H",  # H2
    "H",  # H3
    "H",  # H4
    "H",  # H5
    "H",  # H6
]

# --------------------------
# xyz-positioner (förenklad geometri, ej exakt)
positions = [
    Point3f(0.0, 0.0, 0.0),        # C1
    Point3f(1.5, 0.0, 0.0),        # C2
    Point3f(2.1, 0.9, 0.0),        # O
    Point3f(-0.6, 0.9, 0.0),       # H1 (på C1)
    Point3f(-0.6, -0.9, 0.0),      # H2
    Point3f(0.0, 0.0, 1.0),        # H3
    Point3f(1.5, 0.0, -1.0),       # H4 (på C2)
    Point3f(2.1, -0.9, 0.0),       # H5
    Point3f(2.7, 0.9, 0.0),        # H6 (på O)
]

# --------------------------
# Färger och radier
atom_colors = Dict("H" => :white, "C" => :black, "O" => :red, "N" => :blue)
atom_radii = Dict("H" => 0.2, "C" => 0.35, "O" => 0.4, "N" => 0.4)

# --------------------------
# Hitta bindningar
function find_bonds(positions; cutoff=1.2)
    bonds = Tuple{Int, Int}[]
    for i in 1:length(positions)-1
        for j in i+1:length(positions)
            dist = norm(positions[i] - positions[j])
            if dist ≤ cutoff
                push!(bonds, (i, j))
            end
        end
    end
    return bonds
end

bonds = find_bonds(positions)

# --------------------------
# Visualisering
set_theme!(theme_black())
fig = Figure(resolution = (800, 600))
ax = Axis3(fig[1, 1], aspect = :data)

# Rita atomer som sfärer
for (i, pos) in enumerate(positions)
    atom = atom_types[i]
    color = atom_colors[atom]
    radius = atom_radii[atom]
    mesh!(Sphere(pos, radius), color = color)
end

# Rita bindningar som linjer
for (i, j) in bonds
    p1 = positions[i]
    p2 = positions[j]
    lines!(Point3f[p1, p2], color = :gray, linewidth = 2)
end

fig
