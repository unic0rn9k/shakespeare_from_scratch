struct Linear
    W::NodeID
    b::NodeID
    out::NodeID

    function Linear(g::ADGraph, input_size::Int, output_size::Int, input::NodeID; bias::Bool=true)
        W = rand(g, (input_size, output_size))
        b = rand(g, (1, output_size))
        out = if bias
            input * W + b
        else
            input * W
        end
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

    function AttentionHead(g::ADGraph, q::NodeID, k::NodeID, v::NodeID, embd::Int, headd::Int, mask::Bool)
        seqd = size(val(q))[1]
        @assert seqd == size(val(k))[1] == size(val(v))[1]
        @assert embd == size(val(q))[2] == size(val(k))[2] == size(val(v))[2]

        Wq = Linear(g, embd, headd, q, bias=false)
        Wk = Linear(g, embd, headd, k, bias=false)
        Wv = Linear(g, embd, headd, v, bias=false)
        scores = Wq.out * transpose(Wk.out) / push!(g, sqrt(headd))

        if mask
            tril = push!(g, [r >= c ? 0 : -Inf for r in 1:seqd, c in 1:seqd])
            scores = scores + tril
        end

        attn = softmax(scores) * Wv.out

        rename!(Wk.out, "Wk")
        rename!(Wq.out, "Wq")
        rename!(Wv.out, "Wv")
        rename!(attn, "attn")
        rename!(Wo.out, "Wo")

        return new(de, dk, dv, Wq, Wk, Wv, Wo, attn)
    end
end

## Train single attention head on tiny-shakespeare, with cross_entropy loss and Adam optimizer
# 1. Load data

data = open("tiny-shakespeare.txt") do f
    read(f, String)
end

alphabet = unique(data)
alphabet_size = length(alphabet)
char_to_idx = Dict(ch => i for (i, ch) in enumerate(alphabet))
idx_to_char = Dict(i => ch for (i, ch) in enumerate(alphabet))

data = [char_to_idx[ch] for ch in data]

function get_sequence(seq_len::Int)::Tuple{Matrix{Float64},Matrix{Float64}}
    x = zeros(seq_len, alphabet_size)
    y = zeros(seq_len, alphabet_size)
end
