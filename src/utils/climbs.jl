# CLIMBS FUNCTION ####
# If the atom is by itself and has no () = 0 
# If there is one atom inside the ()= 1
# If there are more than one inside the (), then the last atom will recieve a climb of the amount of atoms in that chain but the other ones will have zero (except if there are more sidechains in that sidechain).
# The last atom in the molecule always get a climb 0 
clean_smiles(smiles::String) = filter(c -> (isletter(c) && c != 'H') || c in ('(', ')'), uppercase(smiles))

function smiles_to_climbs(smiles::String)
    smiles = clean_smiles(smiles)
    stack = Int[]
    next_gid = 1
    direct_count = Dict{Int,Int}()
    last_atom = Dict{Int,Int}()
    atom_direct_group = Int[]
    climbs = Int[]
    atom_idx = 0
    for c in smiles
        if c == '('
            push!(stack, next_gid)
            direct_count[next_gid] = 0
            last_atom[next_gid]   = 0
            next_gid += 1
        elseif c == ')'
            gid = pop!(stack)
            li = last_atom[gid]
            if li == 0
                continue
            end
            nested_len = (atom_direct_group[li] == gid) ? 0 : climbs[li]
            path_len = direct_count[gid] + nested_len
            climbs[li] = max(climbs[li], path_len)
        elseif isletter(c)
            atom_idx += 1
            push!(climbs, 0)
            gid_top = isempty(stack) ? 0 : stack[end]
            push!(atom_direct_group, gid_top)
            if gid_top != 0
                direct_count[gid_top] += 1
            end
            for gid in stack
                last_atom[gid] = atom_idx
            end
        end
    end

    return climbs[2:end]
end

"""
    climbs_to_pairs(climbs) -> Vector{Pair{Int,Int}}

Given a climb value for every atom (as produced by `smiles_to_climbs`),
return an *ordered* collection of directed edges that describes the tree /
branching structure.

Rule implemented
----------------
1.  Start at atom 1 (the “current parent”).
2.  For every entry `i` in `climbs`
       • create the next atom `child = i + 1`  
       • add the edge `current_parent => child`  
       • walk **up** the tree `climbs[i]` steps  
         (never past the root) and make the node we land on
         the new `current_parent`
3.  After the loop, if the node we are currently sitting on is *not* the
    last atom, add one final edge from it to the last atom
    (this closes the ring in cases such as `[0,0,1,0,4]`).

The routine never creates duplicate edges and preserves the order in
which edges are discovered.
"""
function climbs_to_pairs(climbs::AbstractVector{<:Integer})
    n = length(climbs)
    edges = Vector{Pair{Int,Int}}()

    # parent lookup so we can climb back up quickly
    parent = Dict{Int,Int}()

    current = 1                       # we start building from atom 1
    for i in 1:n
        child = i + 1                 # the next atom that will be created
        push!(edges, current => child)
        parent[child] = current       # remember its parent

        # walk 'climbs[i]' steps back up the tree
        steps = climbs[i]
        new_current = child
        while steps > 0 && new_current != 1
            new_current = parent[new_current]
            steps -= 1
        end
        current = new_current
    end

    # If we ended somewhere above the last atom, close the ring.
    last_atom = n + 1
    if current != last_atom
        edge = current => last_atom
        # avoid accidental duplicates
        if !(edge in edges)
            push!(edges, edge)
        end
    end

    return edges
end
