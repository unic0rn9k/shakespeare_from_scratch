begin
    #using Markdown
    #using InteractiveUtils
    #using StaticArrays
    #using CUDA
end

## AD from scratch
# - Caching
# - Pruning
# - Generic type for NodeID
# - Transformers

MathObj = Union{AbstractArray,Number,Nothing}

begin
    Base.:*(::MathObj, ::Nothing)::Nothing = nothing
    Base.:*(::Nothing, ::MathObj)::Nothing = nothing
    Base.:+(l::MathObj, ::Nothing)::MathObj = l
    Base.:+(::Nothing, r::MathObj)::MathObj = r
    Base.:transpose(n::Nothing) = n
    Base.:*(::Nothing, ::Nothing) = nothing
    Base.:+(::Nothing, ::Nothing) = nothing
    Base.:exp(x::AbstractArray) = exp.(x)
    Base.:size(::Nothing) = 0
    Base.:-(::Nothing, ::Nothing) = nothing
    Base.:-(l::MathObj, ::Nothing) = l
    Base.:-(::Nothing, r::MathObj) = -r
    Base.:/(::Nothing, ::Nothing) = nothing
    Base.:/(l::MathObj, ::Nothing) = l
    Base.:/(::Nothing, r::MathObj) = 1 / r
    #Base.:*(l::MathObj, r::MathObj) = l .* r
    #Base.:+(l::MathObj, r::MathObj) = l .+ r
    #Base.:-(l::MathObj, r::MathObj) = l .- r
    #Base.:/(l::MathObj, r::MathObj) = l ./ r

    elemop(f::Function, l::MathObj, r::MathObj)::MathObj = f.(l, r)
    elemop(f::Function, l::MathObj, r::Nothing)::MathObj = f(l, r)
    elemop(f::Function, l::Nothing, r::MathObj)::MathObj = f(l, r)
    elemop(::Function, ::Nothing, ::Nothing)::Nothing = nothing
end

struct Operation
    eval::Function
    backwards::Function
end

NodeHash = Tuple{String,Vector{UInt}}

abstract type Graph end

struct NodeID
    id::UInt
    source::Graph
end

mutable struct ADNode
    version::UInt
    name::String
    op::Operation
    inputs::Vector{NodeID}
    function ADNode(a,b,c)
        new(0,a,b,c)
    end
end

nodehash(n::ADNode)::Tuple{String,Vector{UInt}} = (n.name, [i.id for i in n.inputs])

mutable struct ADGraph <: Graph
    version::UInt
    nodes::Vector{ADNode}
    #cache::Dict{Tuple{String,Vector{UInt}},UInt}
    ADGraph() = new(0, [])#, Dict())
    #blibblob::Bool # cache validation blibblob
    # when graph is mutated -> blibblob ≠ blibblob
    # val(graph, node) -> if graph.blibblob = node.blibblob ; return node.cache ;
    # else evaluation... ; node.blibblob = graph.blibblob end
end

function val(node_::NodeID; debug::Bool=false)::MathObj
    g = node_.source
    node = g.nodes[node_.id]
    args = [val(i, debug=debug) for i in node.inputs]
    v = try
        node.op.eval(args)
    catch e
        @warn("[$(node_.id)]\t $(node.name) : $([size(arg) for arg in args]) = $e")
        rethrow(e)
    end
    if debug
        @info("[$(node_.id)]\t $(node.name) : $([size(arg) for arg in args]) = $(size(v))")
    end
    v
end

struct DiffCtx
    outerd::NodeID
    wrt::NodeID
end

but(ctx::DiffCtx, d::NodeID)::DiffCtx = DiffCtx(d, ctx.wrt)

function Base.:(==)(::ADNode, ::ADNode)::Bool
    throw("Comparing ADNodes")
end

function as_node(value)::ADNode
    if typeof(value) == ADNode
        value
    else
        ADNode(
            "const $value",
            Operation(
                function (_)
                    value
                end,
                function (g, _)
                    push!(g, nothing)
                end
            ),
            [],
        )
    end
end

# TODO: Fix pruning 
function Base.:push!(g::ADGraph, node)::NodeID
    if typeof(node) == NodeID
        throw("Cannot push NodeID to Graph")
    end
    data = as_node(node)
    #nh = nodehash(node)
    #if occursin("const", nh[1])
    #    nh = ("const $(length(g.nodes)))", [])
    #end
    #if nh[1] != "const" && haskey(g.cache, nh)
    #    return NodeID(g.cache[nh], g)
    #end
    push!(g.nodes, data)
    NodeID(length(g.nodes), g)
end

function Base.:rand(g::ADGraph, shape::Tuple{Vararg{Int}})::NodeID
    push!(g, rand(shape...) .* 2 .- 1)
end

function set!(node::NodeID, value)
    node.source.nodes[node.id] = as_node(value)
    node.source.nodes[node.id].version = node.source.version
end

function rename!(node::NodeID, name::String)
    node.source.nodes[node.id].name = name
end

function →(node::NodeID)::ADNode
    adnode = node.source.nodes[node.id]
    ADNode(
        "→($(node.id))",
        Operation(
            (x) -> adnode.op.eval(x),
            (g, ctx) -> Δ!(node, ctx)
        ),
        adnode.inputs,
    )
end

wrt(wrt::NodeID) = DiffCtx(push!(wrt.source, 1), wrt)

function Δ!(node::NodeID, ctx::DiffCtx)::NodeID
    if node == ctx.wrt
        ctx.outerd
    else
        g = node.source
        g.nodes[node.id].op.backwards(g, ctx)
    end
end

function Base.:transpose(x::NodeID)::NodeID
    push!(x.source, ADNode(
        "T",
        Operation(
            (x) -> transpose(x[1]),
            (g, ctx) -> Δ!(x, but(ctx, transpose(ctx.outerd)))
        ),
        [x],
    ))
end

function Base.:sum(x::NodeID; dims=1:ndims(val(x)))::NodeID
    push!(x.source, ADNode(
        "sum",
        Operation(
            #(x) -> sum(x[1], dims=dims),
            (x) -> sum(x[1]),
            (g, ctx) -> Δ!(x, ctx)
        ),
        [x],
    ))
end

function Base.:+(a::NodeID, b::NodeID)::NodeID
    @assert(a.source == b.source)
    push!(a.source, ADNode(
        "+",
        Operation(
            x -> elemop(+, x[1], x[2]),
            (g, ctx) -> Δ!(a, ctx) + Δ!(b, ctx)
        ),
        [a, b],
    ))
end

function Base.:*(a::NodeID, b::NodeID)::NodeID
    @assert(a.source == b.source)
    push!(a.source, ADNode(
        "*",
        Operation(
            prod,
            function (g, ctx)
                bo = ctx.outerd * transpose(b)
                ao = transpose(a) * ctx.outerd

                da = Δ!(a, but(ctx, bo))
                db = Δ!(b, but(ctx, ao))
                da + db
            end
        ),
        [a, b],
    ))
end

function elemmul(a::NodeID, b::NodeID)::NodeID
    @assert(a.source == b.source)
    push!(a.source, ADNode(
        ".*",
        Operation(
            x -> elemop(*, x[1], x[2]),
            function (g, ctx)
                da = Δ!(a, but(ctx, elemmul(ctx.outerd, b)))
                db = Δ!(b, but(ctx, elemmul(ctx.outerd, a)))
                da + db
            end
        ),
        [a, b],
    ))
end

function Base.:exp(x::NodeID)::NodeID
    push!(x.source, ADNode(
        "exp",
        Operation(
            (x) -> exp.(x[1]),
            function (g, ctx)
                Δ!(x, but(ctx, elemmul(exp(x), ctx.outerd)))
            end
        ),
        [x],
    ))
end

function Base.:-(a::NodeID, b::NodeID)::NodeID
    @assert(a.source == b.source)
    push!(a.source, ADNode(
        "-",
        Operation(
            (x) -> elemop(-, x[1], x[2]),
            (g, ctx) -> begin
                #@show val(ctx.outerd)
                #@show Δ!(a, ctx)
                #@show -Δ!(b, ctx)
                Δ!(a, ctx) + Δ!(b, but(ctx, -ctx.outerd))
            end
        ),
        [a, b],
    ))
end

function Base.:-(a::NodeID)::NodeID
    push!(a.source, ADNode(
        "neg",
        Operation(
            (x) -> -x[1],
            (g, ctx) -> Δ!(a, but(ctx, -ctx.outerd))
        ),
        [a],
    ))
end

function Base.:/(a::NodeID, b::NodeID)::NodeID
    @assert(a.source == b.source)
    push!(a.source, ADNode(
        "./",
        Operation(
            x -> elemop(/, x[1], x[2]),
            function (g, ctx)
                ao = ctx.outerd / b
                bo = elemmul(a / elemmul(b, b), ctx.outerd)

                da = Δ!(a, but(ctx, ao))
                db = Δ!(b, but(ctx, bo))
                da - db
            end
        ),
        [a, b],
    ))
end

function Base.:^(x::NodeID, n::Integer)::NodeID
    push!(x.source, ADNode(
        "^$n",
        Operation(
            (x) -> elemop(^, x[1], n),
            function (g, ctx)
                Δ!(x, but(ctx, ctx.outerd * elemmul(push!(g, n), x^(n - 1))))
                #Δ!(x, ctx) * push!(g, n) * ctx.outerd * x^(n - 1)
            end
        ),
        [x],
    ))
end

function Base.:log(x::NodeID)::NodeID
    push!(x.source, ADNode(
        "log",
        Operation(
            (x) -> log.(x[1]),
            function (g, ctx)
                Δ!(x, but(ctx, ctx.outerd / x))
            end
        ),
        [x],
    ))
end

function padded(x::NodeID, size::Tuple, pos::Tuple)::NodeID
    push!(x.source, ADNode(
        "padded $size $pos",
        Operation(
            function (x)
                ret = zeros(size)
                ret[pos...] = x[1]
                return ret
            end,
            function (g, ctx)
                @error("Padding derivative not implemented")
            end
        ),
        [x],
    ))
end

function Base.:getindex(x::NodeID, i...)::NodeID
    push!(x.source, ADNode(
        "getindex $i",
        Operation(
            (x) -> getindex(x[1], i...),
            function (g, ctx)
                Δ!(x, but(ctx, padded(ctx.outerd, size(val(x)), i)))
            end
        ),
        [x],
    ))
end

function Base.:cat(nodes::NodeID...; dims::Int)::NodeID
    push!(nodes[1].source, ADNode(
        "cat dims=$dims",
        Operation(
            (x) -> cat(filter(x -> x !== nothing, x)..., dims=dims),
            function (g, ctx)
                outerd::Vector{Any} = [nothing for _ in nodes]
                i = 1
                for (n, node) in enumerate(nodes)
                    s = [1:n for n in size(val(node))]
                    j = s[dims].stop
                    s[dims] = (i):(j+i-1)
                    outerd[n] = ctx.outerd[s...]
                    i += j
                end
                s = [1:n for n in size(val(ctx.wrt))]
                n = length(s) + 1
                sum(cat([Δ!(node, but(ctx, d)) for (node, d) in zip(nodes, outerd)]..., dims=n), dims=n)[s..., 1]
            end
        ),
        [nodes...],
    ))
end

# From pytorch sourcecode: https://github.com/pytorch/pytorch/blob/9e72c9cccd57bde2d8020c434649a63c3ab0139e/aten/src/ATen/native/transformers/cuda/flash_attn/softmax.h#L98
# Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
# max * log_2(e)) This allows the compiler to use the ffma
# instruction instead of fadd and fmul separately.
function softmax(x::MathObj)
    sm = exp2.(x .* log2(ℯ) .- maximum(x) .* log2(ℯ))
    return sm ./ sum(sm)
end

function softmax(x::NodeID)::NodeID
    push!(x.source, ADNode(
        "softmax",
        Operation(
            (x) -> softmax(x[1]),
            function (g, ctx)
                Δ!(x, but(ctx, elemmul(elemmul(softmax(x), (push!(g, 1) - softmax(x))), ctx.outerd)))
            end
        ),
        [x],
    ))
end

# Implement debugging nodes, as single line equations, without values
function Base.:show(io::IO, node::NodeID)
    inner = node.source.nodes[node.id]
    if occursin("const", inner.name)
        v = val(node)
        if v === nothing
            print(io, 0)
        else
            print(io, inner.name)
        end
    elseif inner.name in ["+", "-", "*", "/", ".*", "./"]
        print(io,
            "(", inner.inputs[1], " ", inner.name,
            " ", inner.inputs[2], ")")
    else
        print(io, "$(inner.name)(")
        for i in inner.inputs
            print(io, " $i")
        end
        print(io, ")")
    end
end

function query_node(g::ADGraph, name::String)::NodeID
    for (i, node) in enumerate(g.nodes)
        if node.name == name
            return NodeID(i, g)
        end
    end
    throw("Node not found")
end

function numeric_gradient(node::NodeID, wrt::Vector{NodeID}; delta=1e-3, iters=100)::Vector{MathObj}
    grads = [zeros(size(val(i))) for i in wrt]
    y = val(node)

    for _ in 1:iters
        for (n, i) in enumerate(wrt)
            input = deepcopy(i)
            dx = randn(size(val(input))) * delta
            set!(input, val(input) + dx)

            dy = val(NodeID(node.id, input.source)) - y
            for dy in dy
                grads[n] += dy ./ dx
            end

            grads[n] /= length(dy)
        end
    end

    [grad / iters for grad in grads]
end

function Δ(f, wrt; cuda=false)
    v = val(f)
    od = if typeof(v) <: Number
        1
    elseif cuda
        os = CUDA.ones(size(v))
    else
        os = ones(size(v))
    end
    Δ!(f, DiffCtx(push!(f.source, od), wrt))
end

## Unit tests
@testset "autodiff.jl" begin

@testset "Basic scalar AD" begin
    local g = ADGraph()
    local a = push!(g, 3)

    local b = push!(g, 4)
    local c = a * b
    @test(val(a) == 3)
    @test(val(c) == 3 * 4)
    local db = Δ!(c, wrt(b))
    @test(val(db) == 3)

    local d = push!(g, 5)
    local e = c + d
    local f = e * push!(g, 2)

    @test(val(Δ(f, d)) == 2)
    @test(val(Δ(f, c)) == 2)
    @test(val(Δ(f, b)) == 6)
end

@testset "Basic matrix tests + (mby) CUDA" begin
    local g = ADGraph()
    #local a = transpose(push!(g, CuArray([1 2; 3 4; 5 6])))
    #local b = push!(g, transpose(CuArray([1 2 3 0; 4 5 6 0; 7 8 9 0])))
    #local d = push!(g, CUDA.ones(2, 4))

    local a = transpose(push!(g, [1 2; 3 4; 5 6]))
    local b = push!(g, transpose([1 2 3 0; 4 5 6 0; 7 8 9 0]))
    local d = push!(g, ones(2, 4))

    local c = exp(d) + a * transpose(b)
    local c = c * b + a

    local da = val(Δ(c, a))
    local db = val(Δ(c, b))
    local dd = val(Δ(c, d))
    #gn = length(g.nodes)
    #local bruh = c*b
    #@assert(gn == length(g.nodes), keys(g.cache))

    @test(size(da) == size(val(a)))
    @test(size(db) == size(val(b)))
    @test(size(dd) == size(val(d)))

    #@test dd ≈ numeric_gradient(c, [d])[1]
end

@testset "Softmax test" begin
    local x = rand(1000)
    local sm = exp.(x) / sum(exp.(x))
    local dsm = sm .* (1 .- sm)

    local g = ADGraph()
    local x = push!(g, x)

    local sm2 = softmax(x)
    local dsm2 = Δ(sm2, x)

    @test(sm ≈ val(sm2))
    @test(dsm ≈ val(dsm2))
end

@testset "cat and slice test" begin
    #local g = ADGraph()
    #local a = rand(g, (3, 4))
    #local b = rand(g, (3, 4))

    #local c = cat(a, elemmul(b, a), dims=2)
    #local d = c[1:3, 3:5]

    #local db = Δ(c, b)
    #local da = Δ(d, a)

    #@test(val(db) == val(a))
    ##println(val(da, debug=true))
    #e = cat(a, push!(g, nothing), dims=1)
    #@test val(e) == val(a)
end
end
