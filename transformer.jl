include("autodiff.jl")

struct Linear
    W::NodeID
    b::NodeID
    out::NodeID

    function Linear(g::ADGraph, input_size::Int, output_size::Int, input::NodeID)
        W = rand(g, (input_size, output_size))
        b = rand(g, (1, output_size))
        out = input * W + b
        return new(W, b, out)
    end
end

struct AttentionHead
    de::UInt
    dk::UInt
    dv::UInt
    Wq::Linear
    Wk::Linear
    Wv::Linear
    Wo::Linear
    out::NodeID

    function AttentionHead(g::ADGraph, de::Int, q::NodeID, k::NodeID, v::NodeID; dk::Int=de, dv::Int=de)
        @assert size(val(q))[1] == size(val(k))[1]
        @assert size(val(q))[1] == size(val(v))[1]
        @assert size(val(k))[2] == size(val(v))[2]

        Wq = Linear(g, de, dk, q)
        rename!(Wq.out, "Wq")
        Wk = Linear(g, de, dk, k)
        rename!(Wk.out, "Wk")
        Wv = Linear(g, de, dv, v)
        rename!(Wv.out, "Wv")
        attn = softmax(Wq.out * transpose(Wk.out) / push!(g, sqrt(dk))) * Wv.out
        rename!(attn, "attn")
        Wo = Linear(g, dv, de, attn)
        rename!(Wo.out, "Wo")

        return new(de, dk, dv, Wq, Wk, Wv, Wo, Wo.out)
    end
end

testgraph = ADGraph()
i = rand(testgraph, (1, 10))
mask = push!(testgraph, floor.(rand(1, 10)))
target = elemmul(i, mask)
a = AttentionHead(testgraph, 10, i, i, i)
cost = sum(elemmul((a.out - target), (a.out - target)))

dqw = Δ(cost, a.Wq.W)
dqb = Δ(cost, a.Wq.b)
dkw = Δ(cost, a.Wk.W)
dkb = Δ(cost, a.Wk.b)
dvw = Δ(cost, a.Wv.W)
dvb = Δ(cost, a.Wv.b)
dow = Δ(cost, a.Wo.W)
dob = Δ(cost, a.Wo.b)

val(cost)

size(val(dqb, debug=true))
size(val(a.Wq.b))

for _ in 0:2
    set!(a.Wq.W, val(a.Wq.W) - 0.001 * val(dqw))
    set!(a.Wq.b, val(a.Wq.b) - 0.001 * val(dqb))
    set!(a.Wk.W, val(a.Wk.W) - 0.001 * val(dkw))
    set!(a.Wk.b, val(a.Wk.b) - 0.001 * val(dkb))
    set!(a.Wv.W, val(a.Wv.W) - 0.001 * val(dvw))
    set!(a.Wv.b, val(a.Wv.b) - 0.001 * val(dvb))
    set!(a.Wo.W, val(a.Wo.W) - 0.001 * val(dow))
    set!(a.Wo.b, val(a.Wo.b) - 0.001 * val(dob))
end

val(cost)

struct TransformerEncoderLayer
    heads::UInt
    de::UInt
    dk::UInt
    dv::UInt
    Wq::Linear
    Wk::Linear
    Wv::Linear
    Wo::Linear
    out::NodeID

    function TransformerEncoderLayer(g::ADGraph, heads::UInt, de::UInt, dk::UInt, dv::UInt, q::NodeID, k::NodeID, v::NodeID)
        heads = [AttentionHead(g, de, q, k, v; dk=dk, dv=dv) for _ in 1:heads]
        out = vcat([head.out for head in heads]...)
        return new(heads, out)
    end
end

struct MultiHeadAttention
    heads::Vector{AttentionHead}
    out::NodeID

    function MultiHeadAttention(g::ADGraph, heads::UInt, de::UInt, dk::UInt, dv::UInt, q::NodeID, k::NodeID, v::NodeID)
        heads = [AttentionHead(g, de, q, k, v; dk=dk, dv=dv) for _ in 1:heads]
        out = vcat([head.out for head in heads]...)
        return new(heads, out)
    end
end
