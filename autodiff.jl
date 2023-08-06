begin
    using Markdown
    using InteractiveUtils
    using StaticArrays
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
    Base.:size(::Nothing) = nothing
    Base.:-(::Nothing, ::Nothing) = nothing
    Base.:-(l::MathObj, ::Nothing) = l
    Base.:-(::Nothing, r::MathObj) = -r
    Base.:/(::Nothing, ::Nothing) = nothing
    Base.:/(l::MathObj, ::Nothing) = l
    Base.:/(::Nothing, r::MathObj) = 1 / r
    Base.:*(l::MathObj, r::MathObj) = l .* r
    Base.:+(l::MathObj, r::MathObj) = l .+ r
    Base.:-(l::MathObj, r::MathObj) = l .- r
    Base.:/(l::MathObj, r::MathObj) = l ./ r

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
end

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
        @error("[$(node_.id)]\t $(node.name) : $args = $e")
        rethrow(e)
    end
    if debug
        @info("[$(node_.id)]\t $(node.name) : $args = $v")
    end
    v
end

struct DiffCtx
    outerd::NodeID
    wrt::NodeID
end

but(ctx::DiffCtx, d::NodeID)::DiffCtx = DiffCtx(d, ctx.wrt)

function Base.:(==)(a::ADNode, b::ADNode)::Bool
    a.name == b.name && a.inputs == b.inputs
end

function as_node(value)::ADNode
    if typeof(value) == ADNode
        value
    else
        ADNode(
            "const($value)",
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

function Base.:push!(g::ADGraph, node)::NodeID
    if typeof(node) == NodeID
        throw("Cannot push NodeID to Graph")
    end
    node = as_node(node)
    nh = nodehash(node)
    if haskey(g.cache, nh)
        return NodeID(g.cache[nh], g)
    end
    push!(g.nodes, node)
    NodeID(get!(g.cache, nh, length(g.nodes)), g)
end

function Base.:rand(g::ADGraph, shape::Tuple{Vararg{Int}})::NodeID
    push!(g, rand(shape...))
end

function set!(node::NodeID, value)
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

function Base.:sum(x::NodeID)::NodeID
    push!(x.source, ADNode(
        "sum",
        Operation(
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
                bruh = elemmul(exp(x), ctx.outerd)
                Δ!(x, but(ctx, bruh))
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
            (g, ctx) -> Δ!(a, ctx) - Δ!(b, ctx)
        ),
        [a, b],
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

softmax(x) = exp(x) / sum(exp(x))

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

begin # Basic scalar tests
    local g = ADGraph()
    local a = push!(g, 3)

    local b = push!(g, 4)
    local c = a * b
    @assert(val(a) == 3, val(a))
    @assert(val(c) == 3 * 4, val(c, debug=true))
    local db = Δ!(c, wrt(b))
    @assert(val(db) == 3, val(db))

    local d = push!(g, 5)
    local e = c + d
    local f = e * push!(g, 2)

    @assert(val(Δ(f, d)) == 2, "$(val(Δ(f, d)))")
    @assert(val(Δ(f, c)) == 2)
    @assert(val(Δ(f, b)) == 6)
end

begin # Basic matrix tests + CUDA
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

    @assert(size(da) == size(val(a)), "$(size(da)) == $(size(val(a)))")
    @assert(size(db) == size(val(b)), "$(size(db)) == $(size(val(b)))")
    @assert(size(dd) == size(val(d)), "$(size(dd)) == $(size(val(d)))")
end

begin # Softmax test
    local x = rand(1000)
    local sm = softmax(x)
    local dsm = sm .* (1 .- sm)

    local g = ADGraph()
    local x = push!(g, x)

    local sm2 = softmax(x)
    local dsm2 = Δ(sm2, x)

    @assert(sm == val(sm2))
    @assert(dsm ≈ val(dsm2), "$(dsm) ≈ $(val(dsm2))")
end