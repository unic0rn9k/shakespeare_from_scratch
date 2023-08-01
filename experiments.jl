mutable struct DenseLayer
    w::Matrix{Float64}
    b::Vector{Float64}
    a::Function
end

function rand_net(i, o, a)::DenseLayer
    DenseLayer(rand(o,i)*2 .- 1, rand(o)*2 .- 1, a)
end

function predict(self::DenseLayer, input::Vector{Float64})::Vector{Float64}
    self.a(self.w * input + self.b)
end

function softmax(x::Vector{Float64})
    exps = exp.(x .- maximum(x))
    exps / sum(exps)
end

l1 = rand_net(200,40,(x->tanh.(x)))
l2 = rand_net(40,20, softmax)

test = rand(20)
println(sum((predict(l2, predict(l1, vcat(rand(180),test)))-test).^2))

batch_size = 32
learning_rate = 0.0001
num_iterations = 8000

for i in 0:num_iterations
    # Initialize gradients
    grad_l1_w = zeros(size(l1.w))
    grad_l1_b = zeros(size(l1.b))
    grad_l2_w = zeros(size(l2.w))
    grad_l2_b = zeros(size(l2.b))

    # Mini-batch training
    for _ in 1:batch_size
        label = zeros(20)
        label[rand(UInt)%length(label)+1]
        input = vcat(rand(180)*2 .- 1, label)

        l1o = predict(l1, input)
        l2o = predict(l2, l1o)

        d2 = -label .* (1 .- l2o)
        grad_l2_w += d2 * transpose(l1o)
        grad_l2_b += d2

        d1 = transpose(l2.w) * d2
        d1 = d1 .* (-l1o.^2 .+ 1)
        grad_l1_w += d1 * transpose(input)
        grad_l1_b += d1
    end

    # Update weights and biases
    l2.w -= (grad_l2_w / batch_size) * learning_rate
    l2.b -= (grad_l2_b / batch_size) * learning_rate
    l1.w -= (grad_l1_w / batch_size) * learning_rate
    l1.b -= (grad_l1_b / batch_size) * learning_rate
end

println(sum((predict(l2, predict(l1, vcat(rand(180),test)))-test).^2))
