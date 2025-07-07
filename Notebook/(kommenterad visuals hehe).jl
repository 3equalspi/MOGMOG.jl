using Pkg
Pkg.activate("visuals", shared=true)

using GLMakie
using LinearAlgebra

# --------------------------
# Exempeldata: atomtyper och xyz koordinater
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
atom_colors = Dict("H" => :white, "C" => :purple, "O" => :red, "N" => :blue) # skapar en dictionary där varje atomtyp får en färg
atom_radii = Dict("H" => 0.2, "C" => 0.35, "O" => 0.4, "N" => 0.4)# skapar en dictionary där varje atomtyp får en radie 

# --------------------------
# Bindningar: koppla atomer inom viss distans
function find_bonds(positions; cutoff=1.2) # hittar atomer inom avstånd 1.2 
    bonds = Tuple{Int, Int}[] # skapar en tom array som ska fyllas med par där (i,j) är bundna
    for i in 1:length(positions)-1 # yttre loop som går från första till näst sista atomen 
        for j in i+1:length(positions)# inre loop som går från efter 1 till sista atomen. Dessa två loopar körs så att atomerna jämförs med varandra utan upprepning. Par skapas. 
            dist = norm(positions[i] - positions[j]) # tar positionerna minus varandra för att få vektoravståndet. Norm beräknar avståndet i 3D
            if dist ≤ cutoff # om atomerna är inom cutoff 
                push!(bonds, (i, j)) # atomparet läggs till i arrayn 
            end
        end
    end
    return bonds
end

bonds = find_bonds(positions)

# --------------------------
# Visualisering
set_theme!(theme_black()) # svart tema är snyggt 
fig = Figure(resolution = (800, 600)) # Skapar en figur med storlek 800×600 pixlar.
ax = Axis3(fig[1, 1], aspect = :data) # Skapar en 3D-axel i figurens första ruta.

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