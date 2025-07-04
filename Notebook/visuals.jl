using Pkg
Pkg.activate("visuals", shared=true)

using GLMakie
using LinearAlgebra

# --------------------------
# Exempeldata: atomtyper och koordinater
atom_types = ["C", "H", "H", "H", "H"]  # T.ex. metan (CH4)
positions = [
    Point3f(0.0, 0.0, 0.0),
    Point3f(0.63, 0.63, 0.63),
    Point3f(-0.63, -0.63, 0.63),
    Point3f(0.63, -0.63, -0.63),
    Point3f(-0.63, 0.63, -0.63),
]

# --------------------------
# Färger och radier för atomer
atom_colors = Dict("H" => :white, "C" => :purple, "O" => :red, "N" => :blue)
atom_radii = Dict("H" => 0.2, "C" => 0.35, "O" => 0.4, "N" => 0.4)

# --------------------------
# Bindningar: koppla atomer inom viss distans
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

# Rita atomer (som sfärer)
for (i, pos) in enumerate(positions)
    atom = atom_types[i]
    color = atom_colors[atom]
    radius = atom_radii[atom]
    mesh!(Sphere(pos, radius), color = color)
end

# Rita bindningar (som linjer mellan atomer)
for (i, j) in bonds
    p1 = positions[i]
    p2 = positions[j]
    lines!(Point3f[p1, p2], color = :gray, linewidth = 2)
end

fig
