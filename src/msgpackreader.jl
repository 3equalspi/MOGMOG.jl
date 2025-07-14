using Tar, CodecZlib, MsgPack
using Printf


struct Molecule
    atoms::Vector{String}
    positions::Matrix{Float64} 
end


function extract_msgpack(tgz_path::String, dest::String="qm9_extracted")
    if isdir(dest)
        rm(dest; force=true, recursive=true)
    end
    mkdir(dest)
    open(tgz_path) do io
        Tar.extract(GzipDecompressorStream(io), dest)
    end
    return filter(fn->endswith(fn, ".msgpack"), readdir(dest; join=true))[1]
end


function read_msgpack(path::String)
    println("Unpacking MsgPack from: $path")
    bytes = open(path) do io
        read(io)
    end
    return MsgPack.unpack(bytes)
end

function atomic_symbol(num)
    table = Dict(1.0=>"H", 6.0=>"C", 7.0=>"N", 8.0=>"O", 9.0=>"F")
    return get(table, num, "?")
end

function convert(raw_data)
    mols = Molecule[]
    for (i, d) in enumerate(raw_data)
        if !(isa(d, AbstractDict) && haskey(d, "conformers"))
            @warn "Entry $i missing 'conformers' key or not a Dict: $(typeof(d)), keys: $(isa(d, AbstractDict) ? keys(d) : "N/A")"
            continue
        end
        conformers = d["conformers"]
        if isempty(conformers)
            @warn "Entry $i has no conformers"
            continue
        end
        conf = conformers[1]
        xyz = conf["xyz"]
        atoms = [atomic_symbol(row[1]) for row in xyz]
        coords = hcat([row[2:4] for row in xyz]...)'  
        push!(mols, Molecule(atoms, coords))
    end
    return mols
end

function preview_molecules(mols::Vector{Molecule}, N::Int=50)
    n = min(N, length(mols))
    @info "Previewing first $n molecules"
    for i in 1:n
        mol = mols[i]
        @printf("\nMolecule #%d: %d atoms\n", i, length(mol.atoms))
        @printf("Atoms: %s%s\n",
            join(mol.atoms[1:min(10,end)], ", "),
            length(mol.atoms) > 10 ? ", ..." : "")
        @printf("Coordinates (first 3):\n%s\n",
                mol.positions[1:min(3,end), :])
    end
end



function main()
    msgpack_path = extract_msgpack("/home/star/MOGMOG.jl-1/qm9_crude.msgpack.tar.gz")
    raw = read_msgpack(msgpack_path)
    mol_dicts = isa(raw, Dict) ? collect(values(raw)) : raw
    println("First entry type: ", typeof(mol_dicts[1]))
    println("First entry: ", mol_dicts[1])
    if isa(mol_dicts[1], AbstractDict)
        println("First entry keys: ", keys(mol_dicts[1]))
    end
    n_total = length(mol_dicts)
    n_valid = count(d -> haskey(d, "atoms") && haskey(d, "coordinates"), mol_dicts)
    println("Total entries: $n_total, valid molecule entries: $n_valid")
    mols = convert(mol_dicts)
    preview_molecules(mols, 50)
end

main()