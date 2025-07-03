module MOGMOG

using LinearAlgebra

function transform_molecule(X::AbstractMatrix{T}) where T
    @assert size(X, 1) == 3

    Q, _ = qr(randn(T, 3, 3)) # Slumpar en 3*3 matris baserat på normalfördelning. Representeras av rand N. Men för att få en riktig rotation och inte bara kaos siffror måste vi använda QR dekomposition, vilket qr(...) gör. Den delar upp matrisen i 2 delar: Q – en ortogonal matris (tänk: snurr) R – en övre triangulär matris (tänk: skala & sträck. Q,_ innebär strunta i R och tänk på Q. 
    if det(Q) < 0 # Determinanten är ett tal som säger om en matris skalar, vänder eller kollapsar rymden. Determinanten av Q ska vara +1 men om den blir -1 måste vi göra den till +1
        Q[:,1] .= -Q[:,1]
    end
    R = Q # För att kalla Q för R som står för rotation 

    X_rotated = R * X # Först roterar vi varje punkt
    t = randn(T, 3)
    X_transformed = X_rotated .+ t # Sedan translationen vi varje punkt genom att elementvis lägga till translationen. 

    return X_transformed # rotation + translation kallas för transformation 
end

export transform_molecule

include("pdf.jl")
export logpdf_MOG

end
