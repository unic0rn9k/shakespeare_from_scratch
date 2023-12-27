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
    Base.:size(::Nothing) = ()
    Base.:-(::Nothing, ::Nothing) = nothing
    Base.:-(l::MathObj, ::Nothing) = l
    Base.:-(::Nothing, r::MathObj) = -r
    Base.:/(::Nothing, ::Nothing) = nothing
    Base.:/(l::MathObj, ::Nothing) = l
    Base.:/(::Nothing, r::MathObj) = 1 / r

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
    name::String
    op::Operation
    inputs::Vector{NodeID}
    const shape::Tuple
    function ADNode(a,b,c; S::Tuple)
        new(a,b,c,S)
    end
end

Base.:size(node::ADNode) = node.shape
Base.:size(node::NodeID) = size(node.source.nodes[node.id])

nodehash(n::ADNode)::Tuple{String,Vector{UInt}} = (n.name, [i.id for i in n.inputs])

mutable struct ADGraph <: Graph
    nodes::Vector{ADNode}
    cache::Dict{Tuple{String,Vector{UInt}},UInt}
    ADGraph() = new([], Dict())
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
    @assert(size(node) == size(v), "Unexpected shape of computed value.\n... node: $node_\n... expected: $(size(node))\n... found: $(size(v))")
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
    if typeof(value) <: ADNode
        value
    else
        ADNode(
            "???",
            Operation(
                function (_)
                    value
                end,
                function (g, _)
                    push!(g, nothing)
                end
            ),
            [],
            S=size(value)
        )
    end
end

# TODO: Fix pruning 
function Base.:push!(g::ADGraph, node)::NodeID
    if typeof(node) == NodeID
        throw("Cannot push NodeID to Graph")
    end
    data = try
        as_node(node)
    catch
        rethrow(node)
    end
    nh = nodehash(data)
    #if occursin("const", nh[1])
    #    nh = ("const $(length(g.nodes)))", [])
    #end
    if !occursin("const", nh[1]) && haskey(g.cache, nh)
        return NodeID(g.cache[nh], g)
    end
    push!(g.nodes, data)
    NodeID(length(g.nodes), g)
end

function set!(node::NodeID, value)
    @assert(size(value) == size(node), "Cannot write value of new shape to node.\nSetting size of $node\nwith size $(size(node))\nto $(size(value))")
    node.source.nodes[node.id] = as_node(value)
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
        S=size(node)
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

Base.:transpose(x::Tuple{Int, Int}) = (x[2], x[1])
Base.:transpose(::Tuple{}) = ()

function Base.:transpose(x::NodeID)::NodeID
    push!(x.source, ADNode(
        "T",
        Operation(
            (x) -> transpose(x[1]),
            (g, ctx) -> Δ!(x, but(ctx, transpose(ctx.outerd)))
        ),
        [x],
        S=transpose(size(x))
    ))
end

# TODO: Make less stupid
sum_size(x::NodeID, dims) = size(sum(zeros(size(x)), dims=dims))

function Base.:sum(x::NodeID; dims=nothing)::NodeID
    push!(x.source, ADNode(
        "sum(dims=$dims)",
        Operation(
            (x) -> dims===nothing ? sum(x[1]) : sum(x[1], dims=dims),
            (g, ctx) -> Δ!(x, ctx)
        ),
        [x],
        S=dims===nothing ? () : sum_size(x, dims)
    ))
end

function elemop_size_(a::Tuple, b::Tuple)
    for (ax, bx) in zip(a, b)
        if ax != 1 && bx != 1
            @assert(ax == bx, "$a !. $b")
        end
    end
    a
end
elemop_size_(a::Tuple, ::Tuple{}) = a
elemop_size_(::Tuple{}, a::Tuple) = a
elemop_size_(::Tuple{}, ::Tuple{}) = ()
function elemop_size(a::NodeID, b::NodeID)::Tuple
    try
        elemop_size_(size(a), size(b))
    catch
        throw("Dimension mismatch. size($a) != size($b)")
    end
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
        S=elemop_size(a, b)
    ))
end

mul_size_(a::Tuple, ::Tuple{}) = a
mul_size_(::Tuple{}, a::Tuple) = a
mul_size_(::Tuple{}, ::Tuple{}) = ()
function mul_size_(a::Tuple{Int, Int}, b::Tuple{Int, Int})
    @assert a[2] == b[1]
    (a[1], b[2])
end
function mul_size(a::NodeID, b::NodeID)::Tuple
    try
        mul_size_(size(a),size(b))
    catch
        throw("Dimension mismatch. size($a) mm size($b)")
    end
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
        S=mul_size(a, b)
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
        S=elemop_size(a, b)
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
        S=size(x)
    ))
end

function Base.:-(a::NodeID, b::NodeID)::NodeID
    @assert(a.source == b.source)

    push!(a.source, ADNode(
        "-",
        Operation(
            (x) -> elemop(-, x[1], x[2]),
            (g, ctx) -> begin
                Δ!(a, ctx) + Δ!(b, but(ctx, -ctx.outerd))
            end
        ),
        [a, b],
        S=elemop_size(a, b)
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
        S=size(a)
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
        S=elemop_size(a, b)
    ))
end

function Base.:^(x::NodeID, n::Integer)::NodeID
    push!(x.source, ADNode(
        "^$n",
        Operation(
            (x) -> elemop(^, x[1], n),
            function (g, ctx)
                Δ!(x, but(ctx, ctx.outerd * elemmul(push!(g, n), x^(n - 1))))
            end
        ),
        [x],
        S=size(x)
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
        S=size(x)
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
                throw("Padding derivative not implemented")
            end
        ),
        [x],
        S=size
    ))
end

index_size(i::Vector) = (filter(x->x!==(), [index_size(i) for i in i])...,)
index_size(::Int) = ()
index_size(i::UnitRange{Int64}) = i.stop - i.start + 1

function Base.:getindex(x::NodeID, i...)::NodeID
    push!(x.source, ADNode(
        "getindex $i",
        Operation(
            (x) -> getindex(x[1], i...),
            function (g, ctx)
                Δ!(x, but(ctx, padded(ctx.outerd, size(x), i)))
            end
        ),
        [x],
        S=index_size([i...])
    ))
end

# TODO: make less stupid?
# TODO: Check if a value is nothing during evaluation, and then ignore it
function cat_size(sizes::Vector, dims)
    if () in sizes
        @warn("catting () sized object ignored")
    end
    sizes = filter(x->x!==(), sizes)
    size(cat([zeros(s) for s in sizes]..., dims=dims))
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
        S=cat_size([size(n) for n in nodes], dims)
    ))
end

# From pytorch sourcecode: https://github.com/pytorch/pytorch/blob/9e72c9cccd57bde2d8020c434649a63c3ab0139e/aten/src/ATen/native/transformers/cuda/flash_attn/softmax.h#L98
# Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
# max * log_2(e)) This allows the compiler to use the ffma
# instruction instead of fadd and fmul separately.
function softmax(x::MathObj; dims=nothing)
    sm = exp2.(x .* log2(ℯ) .- maximum(x) .* log2(ℯ))
    return sm ./ (dims===nothing ? sum(sm) : sum(sm, dims=dims))
end

function softmax(x::NodeID; dims=nothing)::NodeID
    push!(x.source, ADNode(
        "softmax",
        Operation(
            (x) -> softmax(x[1], dims=dims),
            function (g, ctx)
                Δ!(x, but(ctx, elemmul(elemmul(softmax(x, dims=dims), (push!(g, 1) - softmax(x, dims=dims))), ctx.outerd)))
            end
        ),
        [x],
        S = size(x)
    ))
end

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

function query_node(g::ADGraph, name::String)::Vector{NodeID}
    ret = []
    for (i, node) in enumerate(g.nodes)
        if occursin(name, node.name)
            push!(ret, NodeID(i, g))
        end
    end
    ret
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


function func(f::NodeID, params::NodeID...)::Function
    (vars::MathObj...) -> begin
        for (p, v) in zip(params, vars)
            set!(p, v)
        end
        val(f)
    end
end

Base.:convert(::Type{T}, ::Nothing) where T <: Number = 0

@static if get(ENV, "TEST", 0) == "true"
    using FiniteDiff

    function validate_func(f::NodeID, params::NodeID...)
        for param in params
            dd = convert.(Float64, val(Δ(f, param)))
            x = convert.(Float64, val(param))
            dd2 = sum(FiniteDiff.finite_difference_jacobian(func(f, param), x), dims=1)
            for (a,b) in zip(dd2, dd)
                @test abs(a-b) < 1e-3
            end
        end
    end

    ## Unit tests
    @testset "autodiff.jl" begin
        @testset "Basic scalar AD" begin
            g = ADGraph()
            a = push!(g, 3)

            b = push!(g, 4)
            c = a * b
            @test val(a) == 3
            @test val(c) == 3 * 4
            db = Δ!(c, wrt(b))
            @test val(db) == 3

            d = push!(g, 5)
            e = c + d
            f = e * push!(g, 2)

            @test val(Δ(f, d)) == 2
            @test val(Δ(f, c)) == 2
            @test val(Δ(f, b)) == 6
        end

        @testset "Basic matrix tests + (mby) CUDA" begin
            g = ADGraph()
            #local a = transpose(push!(g, CuArray([1 2; 3 4; 5 6])))
            #local b = push!(g, transpose(CuArray([1 2 3 0; 4 5 6 0; 7 8 9 0])))
            #local d = push!(g, CUDA.ones(2, 4))

            a = transpose(push!(g, [1 2; 3 4; 5 6]))
            b = push!(g, transpose([1 2 3 0; 4 5 6 0; 7 8 9 0]))
            d = push!(g, ones(2, 4))

            c = exp(d) + a * transpose(b)
            c = c * b + a

            da = val(Δ(c, a))
            db = val(Δ(c, b))
            dd = val(Δ(c, d))
            #gn = length(g.nodes)
            #local bruh = c*b
            #@assert(gn == length(g.nodes), keys(g.cache))

            @test size(da) == size(val(a))
            @test size(db) == size(val(b))
            @test size(dd) == size(val(d))

            validate_func(c, a, b, d)
        end

        @testset "Softmax test" begin
            x = rand(1000)
            sm = exp.(x) / sum(exp.(x))
            dsm = sm .* (1 .- sm)

            g = ADGraph()
            x = push!(g, x)

            sm2 = softmax(x)
            dsm2 = Δ(sm2, x)

            @test sm ≈ val(sm2)
            @test dsm ≈ val(dsm2)
        end

        @testset "cat and slice test" begin
            g = ADGraph()
            a = push!(g, randn(3, 4))
            b = push!(g, randn(3, 4))

            c = cat(a, elemmul(b, a), dims=2)
            d = c[1:3, 3:5]

            validate_func(d, a, b, c)
            validate_func(c, a, b)
        end
    end

    @testset "sum" begin
        g = ADGraph()
        a = push!(g, randn(3, 4, 5))
        c = sum(a)
        d = sum(a, dims=1)
        e = sum(a, dims=2)
        f = sum(a, dims=(1,2))

        validate_func(c, a)
        validate_func(d, a)
        validate_func(e, a)
        validate_func(f, a)
    end
end
