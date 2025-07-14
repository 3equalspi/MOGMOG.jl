####OBS######

##TEST, EJ FÄRDIG ALLS

using GLMakie, LinearAlgebra, MOGMOG, Flux, Onion, RandomFeatureMaps, Onion.Einops

include("loss.jl")
include("MOGhead.jl")
include("MOGfoot.jl")
include("model.jl")
include("utilities.jl")

export logpdf_MOG, MoGAxisHead, MOGMOGModel, transform_molecule, loss

embed_dim = 32         # or whatever value you want
n_components = 8       # or your desired value
vocab_size = 6         # matches your atom_dict
depth = 4 

model = MOGMOGModel(embed_dim, n_components, vocab_size, depth=depth)

atom_dict = Dict("C"=>1, "F"=>2, "H"=>3, "N"=>4, "O"=>5, "STOP"=>6)
reverse_atom_dict = Dict(v => k for (k, v) in atom_dict)
atom_colors = Dict("H" => :white, "C" => :gray, "O" => :red, "N" => :blue, "F" => :lightblue)
atom_radii = Dict("H" => 0.2, "C" => 0.35, "O" => 0.4, "N" => 0.4, "F" => 0.4)
PAD = atom_dict["STOP"]
max_len = 32

# === Bindningsfunktion ===
function find_bonds(points; cutoff=1.3)
    bonds = Tuple{Int, Int}[]
    for i in 1:length(points)-1, j in i+1:length(points)
        if norm(points[i] - points[j]) ≤ cutoff
            push!(bonds, (i, j))
        end
    end
    return bonds
end

# === Gör input till batchformat ===
function run_model_on_molecule(model, coords::Matrix{Float64}, atom_ids::Vector{Int})
    L = size(coords, 2)
    coords_padded = zeros(Float32, 3, max_len, 1)
    atom_ids_padded = fill(PAD, max_len, 1)
    coords_padded[:, 1:L, 1] .= Float32.(coords)
    atom_ids_padded[1:L, 1] .= atom_ids
    μ, σ, logw, logits = model(coords_padded, atom_ids_padded)
    return μ[:, :, 1], σ[:, :, 1], logw[:, 1], logits[:, 1]
end

# === Grid-visualisering ===
function visualize_generation_grid(model, mol; max_steps=9, n_rows=3)
    coords = mol.positions'
    atom_ids = [atom_dict[a] for a in mol.atoms]
    n_cols = ceil(Int, max_steps / n_rows)
    fig = Figure(size = (350 * n_cols, 350 * n_rows))

    for step in 1:max_steps
        row = div(step - 1, n_cols) + 1
        col = mod1(step, n_cols)
        ax = Axis3(fig[row, col], aspect=:data)

        curr_coords = coords[:, 1:step]
        curr_atom_ids = atom_ids[1:step]
        atoms = [reverse_atom_dict[i] for i in curr_atom_ids]

        for (i, pos) in enumerate(eachcol(curr_coords))
            atom = atoms[i]
            color = atom_colors[atom]
            radius = atom_radii[atom]
            mesh!(ax, Sphere(Point3f(pos...), radius), color=color)
        end

        bonds = find_bonds([Point3f(c...) for c in eachcol(curr_coords)])
        for (i, j) in bonds
            lines!(ax, Point3f[curr_coords[:, i], curr_coords[:, j]], color=:gray, linewidth=2)
        end

        μ, σ, logw, logits = run_model_on_molecule(model, curr_coords, curr_atom_ids)
        predicted_atom_ids = Flux.onecold(logits; dims=1)
        predicted_atoms = [reverse_atom_dict[i] for i in predicted_atom_ids]

        last_pos = curr_coords[:, end]
        for i in 1:size(μ, 1)
            predicted_pos = last_pos .+ μ[i, :]
            pred_atom = predicted_atoms[i]
            color = get(atom_colors, pred_atom, :black)
            scatter!(ax, [Point3f(predicted_pos...)], marker=:x, color=color, markersize=20)
        end

        hidespines!(ax); hidedecorations!(ax)

        if step < max_steps
            plus_ax = Axis(fig[row, col+1], xticks=[], yticks=[], xvisible=false, yvisible=false)
            text!(plus_ax, "+", position=(0.5, 0.5), align=(:center, :center), fontsize=30)
            hidespines!(plus_ax); hidedecorations!(plus_ax)
        end
    end
end


# Simple test molecule (methanol CH3OH)
positions = [
    [0.0, 0.0, 0.0],    # C
    [0.6, 0.0, 0.0],    # H
    [-0.6, 0.0, 0.0],   # H
    [0.0, 0.9, 0.0],    # H
    [1.0, 0.0, 0.9]     # O
]
atoms = ["C", "H", "H", "H", "O"]
mol = (; positions=reduce(hcat, positions), atoms)

function animate_generation(model, mol; max_steps=5, output="mol.gif")
    coords = mol.positions  # 3 x N
    atom_ids = [atom_dict[a] for a in mol.atoms]

    scene = Scene(size=(400, 400), camera=campixel!)
    ax = Axis3(scene, aspect=:data)

    record(scene, output; framerate=5) do io
        for step in 1:max_steps
            current = coords[:, 1:step]
            @show size(current)
            @show [size(c) for c in eachcol(current)]
            for (i, pos) in enumerate(eachcol(current))
                atom = reverse_atom_dict[atom_ids[i]]
                mesh!(ax, Sphere(Point3f(pos...), atom_radii[atom]), color=atom_colors[atom])
            end
            points = [Point3f(c...) for c in eachcol(current)]
            bonds = find_bonds(points)
            for (i, j) in bonds
                lines!(ax, [points[i], points[j]], color=:gray)
            end
            yield()
            display(scene)
            if step < max_steps
                empty!(ax)
            end
        end
    end
end

animate_generation(nothing, mol; max_steps=5, output="methanol.gif")