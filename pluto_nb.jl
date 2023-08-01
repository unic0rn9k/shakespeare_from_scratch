### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 964c41f8-2a6e-11ee-2f21-b5c944acc6de
begin
    using StaticArrays
	using CUDA
end

# ╔═╡ 408d2c42-a60d-42e5-9d67-95b164707ec7
shakespear = open("/home/unic0rn9k/Documents/shakespear_from_scratch/tiny-shakespeare.txt")

# ╔═╡ 7ec805dc-1697-488e-a272-d908ec3b448c
readline(shakespear)

# ╔═╡ b396a156-e078-4aee-a76e-50315877d42b
tokens::Dict{Char,UInt64} = Dict()

# ╔═╡ 632ff256-1c13-4dc0-88d8-debc281827f2
for c in readeach(shakespear, Char)
    get!(tokens, c, length(tokens))
end

# ╔═╡ 9236be00-058c-4075-ad22-43a25eab2596
Token = SVector{length(tokens),Float64}

# ╔═╡ e16c0261-0cc1-4621-bb0a-89204277fe68
dk = nothing

# ╔═╡ afd05742-58bf-41a6-af45-dcd95d07ba34
block_size = 50

# ╔═╡ dca3db82-e19e-4348-a5d9-c8400601091d
struct Head{S}
    vw#::SMatrix{length(Token), S}
    kw#::SMatrix{length(Token), S}
    qw#::SMatrix{length(Token), S}
end

# ╔═╡ 42662b3e-1112-4bf2-ac52-3a5983e043b8
function random_head(s)::Head{s}
    Head{s}(
        rand(length(Token), s),
        rand(length(Token), s),
        rand(length(Token), s)
    )
end

# ╔═╡ 3ac4517a-d6cd-437b-a12a-9d95c44819ad
h = random_head(10)

# ╔═╡ ba2cb01a-0262-4209-a382-390dd07395f4
md"""
# AD from scratch
- Caching
- Pruning
- Transformers
"""

# ╔═╡ 23d0bb8a-6895-4100-9010-6db0dac84f4f
MathObj = Union{AbstractArray, Number, Nothing}

# ╔═╡ 99f79deb-ac45-4a10-914e-6a39332a6655
begin
	Base.:*(l::MathObj, r::Nothing)::Nothing = nothing
	Base.:*(l::Nothing, r::MathObj)::Nothing = nothing
	Base.:+(l::MathObj, r::Nothing)::MathObj = l
	Base.:+(l::Nothing, r::MathObj)::MathObj = r
	Base.:transpose(n::Nothing) = n
	Base.:*(l::Nothing, r::Nothing) = nothing
	Base.:+(l::Nothing, r::Nothing) = nothing
	Base.:exp(x::AbstractArray) = exp.(x)
	
	elemop(f::Function, l::MathObj, r::MathObj)::MathObj = f.(l, r)
	elemop(f::Function, l::MathObj, r::Nothing)::MathObj = f(l, r)
	elemop(f::Function, l::Nothing, r::MathObj)::MathObj = f(l, r)
	elemop(f::Function, l::Nothing, r::Nothing)::Nothing = nothing
end

# ╔═╡ dceae527-8326-4091-af14-00f9fe7d44fe
struct Operation
    eval::Function
    backwards::Function
end

# ╔═╡ 894f6170-8a5e-4fa8-8a5a-812d77c7c1e6
NodeHash = Tuple{String, Vector{UInt}}

# ╔═╡ 46d31b51-d2f1-43c3-961b-7108524f51ba
abstract type Graph end

# ╔═╡ 82d6644e-fedd-43af-83d6-8f3ec3a081c3
struct NodeID
    id::UInt
    source::Graph
end

# ╔═╡ 4d09fd6b-18d7-455a-8297-79c286a33a0d
struct ADNode
    name::String
    op::Operation
    inputs::Vector{NodeID}
end

# ╔═╡ c6cf5c3d-30c7-4628-97e2-eae6b9811e63
nodehash(n::ADNode)::Tuple{String, Vector{UInt}} = (n.name, [i.id for i in n.inputs])

# ╔═╡ f879e3ec-1c6a-4d46-829f-18c4931db7e3
mutable struct ADGraph <: Graph
    nodes::Vector{ADNode}
	cache::Dict{Tuple{String, Vector{UInt}}, UInt}
    ADGraph() = new([],Dict())
	#blibblob::Bool # cache validation blibblob
	# when graph is mutated -> blibblob ≠ blibblob
	# val(graph, node) -> if graph.blibblob = node.blibblob ; return node.cache ;
	# else evaluation... ; node.blibblob = graph.blibblob end
end

# ╔═╡ 1aefb4d2-7d63-472c-a052-abb6f0d361e2
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

# ╔═╡ 0f41a81b-c0b6-4375-a0a7-d1a141adec7b
struct DiffCtx
    outerd::NodeID
    wrt::NodeID
end

# ╔═╡ d30c4bb2-1c9f-431a-9fdb-1ff117edec50
but(ctx::DiffCtx, d::NodeID)::DiffCtx = DiffCtx(d, ctx.wrt)

# ╔═╡ a1dde7ad-f14d-401b-b38c-ee3963ac6c94
function Base.:(==)(a::ADNode, b::ADNode)::Bool
    a.name == b.name && a.inputs == b.inputs
end

# ╔═╡ d90e79de-f38e-43db-b643-11f4ae678268
function nodeify(value)::ADNode
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

# ╔═╡ 45a314f6-7d4f-4d44-be1b-599c8e6275cc
function Base.push!(g::ADGraph, node)::NodeID
    if typeof(node) == NodeID
        throw("Cannot push NodeID to Graph")
    end
	node = nodeify(node)
	nh = nodehash(node)
	if haskey(g.cache, nh)
		return NodeID(g.cache[nh], g)
	end
    push!(g.nodes, node)
	NodeID(get!(g.cache, nh, length(g.nodes)), g)
end

# ╔═╡ ae75e1d4-f006-48d3-81bf-7a2492a960b6
wrt(wrt::NodeID) = DiffCtx(push!(wrt.source, 1), wrt)

# ╔═╡ 31187841-8d37-438a-ae4d-f74112976d55
function Δ!(node::NodeID, ctx::DiffCtx)::NodeID
    if node == ctx.wrt
        ctx.outerd
    else
		g = node.source
        g.nodes[node.id].op.backwards(g, ctx)
    end
end

# ╔═╡ 3ff66dcf-42a8-419b-9ed0-202c029f1b56
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

# ╔═╡ 6a635a05-cfbc-4af3-8fd4-e3b15e219a66
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

# ╔═╡ ae4b4516-de65-4bc9-a0dd-051a706a61a7
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

# ╔═╡ a85ebb1c-0536-4910-9115-8194933a8a2c
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

# ╔═╡ d894b2c8-2a68-4b19-b0ea-75ec8cd8a978
function predict(head::Head, (v, k, q))::NTuple{3,Matrix}
    (
        q * head.qw,
        k * head.kw,
        v * head.vw
    )
end

# ╔═╡ 5f921837-865e-4236-aad7-73558e62a22e
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

# ╔═╡ 206e199d-7bba-4561-b22c-aa79516a4cbd
function Base.:exp(x::NodeID)::NodeID
	push!(x.source, ADNode(
        "exp",
        Operation(
            (x) -> exp.(x[1]),
            function(g, ctx)
				bruh=elemmul(exp(x), ctx.outerd)
				Δ!(x, but(ctx, bruh))
			end
        ),
        [x],
	))
end

# ╔═╡ b61a5df7-cf61-4962-a9a2-9d289c12d0ee
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

# ╔═╡ a31e338b-9711-47d8-987b-97feaa3487e1
onehot(i, s) = [i == j for j in 0:s-1]

# ╔═╡ 47cb57b5-08fa-41eb-a6d8-9b5a93ef0fdb
token(c::Char)::Token = onehot(UInt64(c), length(Token))

# ╔═╡ 962ae558-c85a-4426-9281-7c73c093f377
function tokenize(str::String)::SMatrix{length(str),length(Token)}
    hcat([token(c) for c in str]...)
end

# ╔═╡ b3eb5c87-e778-4d6b-a8b9-680a70c2ece4
text = tokenize("abcdefg")

# ╔═╡ 34bc575a-8ab7-4939-89f7-7820aefcfd25
predict(h, text)

# ╔═╡ 9216fa15-c85e-4233-9499-abac798645ac
size(text)

# ╔═╡ a80910dc-bd94-4296-a37f-485992e496d5
# qr x S
# qr x qc
# S
function head_backprop!(head::Head, (v, k, q), ∇out)
    (δq, δk, δv) = jacobian(attention, predict(head, (v, k, q)))

    println(typeof(predict(head, (v, k, q))))

    println(size(δq))
    println(size(∇out))
    println(size(q))

    dodq = δq * ∇out'
    head.qw -= (dodq * q)
    return head.qw * (∇out * δq')
end

# ╔═╡ b153cf0a-5d29-4bbc-96ad-4df8d0354e24
function Base.:/(a::NodeID, b::NodeID)::NodeID
	@assert(a.source == b.source)
	push!(a.source, ADNode(
        "./",
        Operation(
            x -> elemop(/, x[1], x[2]),
            function (g, ctx)
				ao = ctx.outerd / b
				bo = elemmul(a / elemmul(b,b), ctx.outerd)
				
                da = Δ!(a, but(ctx, ao))
                db = Δ!(b, but(ctx, bo))
                da - db
            end
        ),
        [a, b],
	))
end

# ╔═╡ a1445527-12fa-4552-8bcd-88e693c4731d
softmax(x) = exp(x) / sum(exp(x))

# ╔═╡ 58e8ed6c-72f7-4539-b88d-99eae0505da7
attn(q, k, v) = softmax(q * k' / 69) * v

# ╔═╡ b1436848-eb7a-43ff-b408-79c1ece34d39
function Δ(f, wrt; cuda=false)
	v = val(f)
	od = if typeof(v)<:Number
		1
	elseif cuda
		os = CUDA.ones(size(v))
	else
		os = ones(size(v))
	end
    Δ!(f, DiffCtx(push!(f.source, od), wrt))
end

# ╔═╡ bf3b3239-0429-4e8d-b670-ea36faa725a3
md"""
# Unit tests
"""

# ╔═╡ 016805bd-8648-4f26-8757-e779d5280154
begin # Basic scalar tests
    local g = ADGraph()
    local a = push!(g, 3)

    local b = push!(g, 4)
    local c = a * b
    @assert(val(a) == 3, val(a))
    @assert(val(c) == 3 * 4, val(c))
    local db = Δ!(c, wrt(b))
    @assert(val(db) == 3, val(db))

    local d = push!(g, 5)
    local e = c + d
    local f = e * push!(g, 2)

    @assert(val(Δ(f, d)) == 2, "$(val(Δ(f, d)))")
    @assert(val(Δ(f, c)) == 2)
    @assert(val(Δ(f, b)) == 6)
end

# ╔═╡ e025b899-bcfc-4d64-8e50-56365474ce80
begin # Basic matrix tests + CUDA
    local g = ADGraph()
    local a = transpose(push!(g, CuArray([1 2; 3 4; 5 6])))
    local b = push!(g, transpose(CuArray([1 2 3 0; 4 5 6 0; 7 8 9 0])))
    local d = push!(g, CUDA.ones(2, 4))
    local c = exp(d) + a * transpose(b)
    local c = c*b + a
	
    local da = val(Δ(c, a, cuda=true))
	local db = val(Δ(c, b, cuda=true))
	local dd = val(Δ(c, d, cuda=true))
	#gn = length(g.nodes)
	#local bruh = c*b
	#@assert(gn == length(g.nodes), keys(g.cache))
	
    @assert(size(da) == size(val(a)), "$(size(da)) == $(size(val(a)))")
	@assert(size(db) == size(val(b)), "$(size(db)) == $(size(val(b)))")
	@assert(size(dd) == size(val(d)), "$(size(dd)) == $(size(val(d)))")
end

# ╔═╡ 42bb37e0-4864-4c75-ac48-f36255d6e1e4
begin # Softmax test
	local x = rand(1000)
	local sm = softmax(x)
	local dsm = sm.*(1 .- sm)
	
	local g = ADGraph()
	local x = push!(g, x)

	local sm2 = softmax(x)
	local dsm2 = Δ(sm2, x)
	
	@assert(sm == val(sm2))
	@assert(dsm ≈ val(dsm2), "$(dsm) ≈ $(val(dsm2))")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
CUDA = "~4.4.0"
StaticArrays = "~1.6.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "f47730ef374f48bd729db37ecc90870b8573934c"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "35160ef0f03b14768abfd68b830f8e3940e8e0dc"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.4.0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "72b2e3c2ba583d1a7aa35129e56cf92e07c083e3"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.21.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4c5875e4c228247e1c2b087669846941fb6e0118"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.8"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "8695a49bfe05a2dc0feeefd06b4ca6361a018729"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.1.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c35203c1e1002747da220ffc3c0762ce7754b08c"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.23+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "9cabadf6e7cd2349b6cf49f1915ad2028d65e881"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.2"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═964c41f8-2a6e-11ee-2f21-b5c944acc6de
# ╠═408d2c42-a60d-42e5-9d67-95b164707ec7
# ╠═7ec805dc-1697-488e-a272-d908ec3b448c
# ╠═b396a156-e078-4aee-a76e-50315877d42b
# ╠═632ff256-1c13-4dc0-88d8-debc281827f2
# ╠═a31e338b-9711-47d8-987b-97feaa3487e1
# ╠═9236be00-058c-4075-ad22-43a25eab2596
# ╠═47cb57b5-08fa-41eb-a6d8-9b5a93ef0fdb
# ╠═962ae558-c85a-4426-9281-7c73c093f377
# ╠═a1445527-12fa-4552-8bcd-88e693c4731d
# ╠═e16c0261-0cc1-4621-bb0a-89204277fe68
# ╠═58e8ed6c-72f7-4539-b88d-99eae0505da7
# ╠═afd05742-58bf-41a6-af45-dcd95d07ba34
# ╠═dca3db82-e19e-4348-a5d9-c8400601091d
# ╠═42662b3e-1112-4bf2-ac52-3a5983e043b8
# ╠═d894b2c8-2a68-4b19-b0ea-75ec8cd8a978
# ╠═a80910dc-bd94-4296-a37f-485992e496d5
# ╠═3ac4517a-d6cd-437b-a12a-9d95c44819ad
# ╠═b3eb5c87-e778-4d6b-a8b9-680a70c2ece4
# ╠═34bc575a-8ab7-4939-89f7-7820aefcfd25
# ╠═9216fa15-c85e-4233-9499-abac798645ac
# ╟─ba2cb01a-0262-4209-a382-390dd07395f4
# ╠═23d0bb8a-6895-4100-9010-6db0dac84f4f
# ╠═99f79deb-ac45-4a10-914e-6a39332a6655
# ╠═dceae527-8326-4091-af14-00f9fe7d44fe
# ╠═4d09fd6b-18d7-455a-8297-79c286a33a0d
# ╠═894f6170-8a5e-4fa8-8a5a-812d77c7c1e6
# ╠═c6cf5c3d-30c7-4628-97e2-eae6b9811e63
# ╠═46d31b51-d2f1-43c3-961b-7108524f51ba
# ╠═f879e3ec-1c6a-4d46-829f-18c4931db7e3
# ╠═82d6644e-fedd-43af-83d6-8f3ec3a081c3
# ╠═d90e79de-f38e-43db-b643-11f4ae678268
# ╠═45a314f6-7d4f-4d44-be1b-599c8e6275cc
# ╠═1aefb4d2-7d63-472c-a052-abb6f0d361e2
# ╠═0f41a81b-c0b6-4375-a0a7-d1a141adec7b
# ╠═31187841-8d37-438a-ae4d-f74112976d55
# ╠═ae75e1d4-f006-48d3-81bf-7a2492a960b6
# ╠═d30c4bb2-1c9f-431a-9fdb-1ff117edec50
# ╠═3ff66dcf-42a8-419b-9ed0-202c029f1b56
# ╠═a85ebb1c-0536-4910-9115-8194933a8a2c
# ╠═b153cf0a-5d29-4bbc-96ad-4df8d0354e24
# ╠═5f921837-865e-4236-aad7-73558e62a22e
# ╠═206e199d-7bba-4561-b22c-aa79516a4cbd
# ╠═ae4b4516-de65-4bc9-a0dd-051a706a61a7
# ╠═6a635a05-cfbc-4af3-8fd4-e3b15e219a66
# ╠═b61a5df7-cf61-4962-a9a2-9d289c12d0ee
# ╠═a1dde7ad-f14d-401b-b38c-ee3963ac6c94
# ╠═b1436848-eb7a-43ff-b408-79c1ece34d39
# ╟─bf3b3239-0429-4e8d-b670-ea36faa725a3
# ╠═016805bd-8648-4f26-8757-e779d5280154
# ╠═e025b899-bcfc-4d64-8e50-56365474ce80
# ╠═42bb37e0-4864-4c75-ac48-f36255d6e1e4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
