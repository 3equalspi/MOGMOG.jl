using GLMakie
using LinearAlgebra

# Atomtyper (Etanol: C2H5OH)
atom_types = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
positions = [
    Point3f(0.0, 0.0, 0.0),
    Point3f(1.5, 0.0, 0.0),
    Point3f(2.1, 0.9, 0.0),
    Point3f(-0.6, 0.9, 0.0),
    Point3f(-0.6, -0.9, 0.0),
    Point3f(0.0, 0.0, 1.0),
    Point3f(1.5, 0.0, -1.0),
    Point3f(2.1, -0.9, 0.0),
    Point3f(2.7, 0.9, 0.0),
]

# --------------------------
# Färger och radier
atom_colors = Dict("H" => :white, "C" => :black, "O" => :red, "N" => :blue)
atom_radii = Dict("H" => 0.2, "C" => 0.35, "O" => 0.4, "N" => 0.4)

# --------------------------
# Hitta bindningar
function find_bonds(positions; cutoff=1.2) # cutoff är max-avståndet för att rita en bindning.
    bonds = Tuple{Int, Int}[] # Skapar en tom lista där varje element är ett par av atomindex (i, j).
    for i in 1:length(positions)-1 # Loopa genom alla möjliga atompar utan att upprepa eller jämföra samma atom med sig själv.
        for j in i+1:length(positions)
            dist = norm(positions[i] - positions[j]) # Beräknar avstånd 
            if dist ≤ cutoff # Om avståndet mellan atomerna är mindre än cutoff ska en bindning ritas 
                push!(bonds, (i, j)) # lägg till paret i listan 
            end
        end
    end
    return bonds
end

bonds = find_bonds(positions) # skickar in positions i funktionen och returnerar bonds, listan med atompar som ska ha en bindning mellan sig 

# === Extra pluspositioner i 3D ===
plus_positions = [
    Point3f(3.0, 0.5, 0.0),  # valfri plats där + ska synas
    Point3f(2.5, 1.5, 0.0), # (point3 betyder att det ritas i en 3D illustration, inte att de ritas 3D)
    Point3f(2.0, 1.5, 0.0),
    Point3f(2.7, 2.0, 0.0),
    Point3f(2.5, 0.5, 0.0),
    
]

# --------------------------
# Visualisering
set_theme!(theme_dark())  
fig = Figure(resolution = (800, 600)) # Skapar en makiefigur 
ax = Axis3(fig[1, 1], aspect = :data) # Lägger till en 3D axel genom axis3

# Rita atomer
for (i, pos) in enumerate(positions) # Loopar över varje atomposition. i är indexet och pos är xyz koordinaterna
    atom = atom_types[i] # hämtar atomtypen 
    color = atom_colors[atom] # hämtar rätt färg 
    radius = atom_radii[atom] # hämtar radien 
    mesh!(Sphere(pos, radius), color = color) # rita en 3D sfär 
end

# Rita bindningar
for (i, j) in bonds # loopa över alla atompar som ska vara bundna 
    lines!(Point3f[positions[i], positions[j]], color = :gray, linewidth = 2) # rita en linje mellan dem 
end

# Rita 2D plustecken
for pos in plus_positions # loopa igenom alla + positioner 
    scatter!(ax, [pos], marker = '+', color = :gray80, markersize = 25, # Här ser man att + ritas i 2D. Tex att + är ett texttecken och blir då i 2D. Scatter används även på platta objekt.  
             transparency = true, fxaa = true)
end

fig
