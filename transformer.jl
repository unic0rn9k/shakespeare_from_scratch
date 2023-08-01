include("autodiff.jl")

struct AttentionHead
    dk::UInt
    dv::UInt
    Wq::NodeId
    Wk::NodeID
    Wv::NodeID
    Wo::NodeID

    function AttentionHead(g, dk, dv)
        Wq
        return new(dk, dv, Wq, Wk, Wv, Wo)
    end
end

attn(q, k, v) = softmax(q * k' / 69) * v

bruh = ADGraph()
some = push!(bruh, 0)
thing = push!(bruh, 1)
set!(some, →(thing))

val(some)