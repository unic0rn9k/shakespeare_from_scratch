function linear(input::NodeID, output_size::Int; bias::Bool=true)
    W = push!(input.source, rand(size(input)[end], output_size))
    b = push!(input.source, rand(1, output_size))
    rename!(W, "some weight param")
    rename!(b, "some bias param")
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

seq_len = 100;

data = [char_to_idx[ch] for ch in data]

function tokenize(x::Vector{Int})::Matrix{Float32}
    [i == j ? 1 : 0 for i in x, j in 1:alphabet_size]
end

function get_sequence()::Tuple{Matrix{Float64},Matrix{Float64}}
    n = rand(1:length(data)-seq_len-1)
    x = data[n:n+seq_len-1]
    y = data[n+1:n+seq_len]
    
    tokenize.(collect.((x, y)))
end

function decoder_block(input::NodeID, nheads::Int; headd::Int=20, outd::Int=alphabet_size)
    heads = [attn(input, input, input, headd, true) for _ in 0:nheads]
    c = cat(heads..., dims=2)
    # seqd x headd*nheads -> outd
    linear(c, outd)
end

g = ADGraph()
x = push!(g, zeros(seq_len, alphabet_size))
y = push!(g, zeros(seq_len, alphabet_size))
rename!(x, "X")
rename!(y, "Y")

bruh = decoder_block(x, 5)
loss = mse_loss(y, bruh)

opt = Adam(0.01, query_node(g, "param"), loss)

for iter in 0:1000
    @show iter

    (a,b) = get_sequence()
    set!(x, a)
    set!(y, b)
    optimize!(opt)

    @show val(loss)
end

@show size(val(bruh))
