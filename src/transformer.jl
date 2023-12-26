function linear(input::NodeID, output_size::Int; bias::Bool=true)
    W = push!(input.source, rand(size(input)[end], output_size))
    b = push!(input.source, rand(1, output_size))
    bias ? input*W+b : input*W
end

function attn(q::NodeID, k::NodeID, v::NodeID, headd::Int, mask::Bool)
    @assert q.source == k.source == v.source
    g=q.source
    seqd = size(q)[1]
    embd = size(q)[2]

    # Doesn't allow for cross attention
    @assert seqd == size(q)[1] == size(k)[1] == size(v)[1]
    @assert embd == size(q)[2] == size(k)[2] == size(v)[2]

    Wq = linear(q, headd, bias=false)
    Wk = linear(k, headd, bias=false)
    Wv = linear(v, headd, bias=false)
    scores = Wq * transpose(Wk) / push!(g, sqrt(headd))

    if mask
        tril = push!(g, [r >= c ? 0 : -Inf for r in 1:seqd, c in 1:seqd])
        rename!(tril, "tril")
        scores = scores + tril
    end

    attn = softmax(scores) * Wv

    rename!(Wk, "Wk")
    rename!(Wq, "Wq")
    rename!(Wv, "Wv")
    rename!(attn, "attn")

    attn
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


    (x, y)
end

function decoder_block(input::NodeID, nheads::Int; headd::Int=20, outd::Int=alphabet_size)
    heads = [attn(input, input, input, headd, true) for _ in 0:nheads]
    c = cat(heads..., dims=2)
    # seqd x headd*nheads -> outd
    linear(c, outd)
end

g = ADGraph()
seq = push!(g, zeros(100, alphabet_size))
bruh = decoder_block(seq, 5)
