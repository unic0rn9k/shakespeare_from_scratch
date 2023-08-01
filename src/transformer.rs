use crate::attention::Attention;

const SMTH: usize = 14;

// RoBERTa says that disentanglement is good
// i  = input       (+ position encoding)
// a  = attention   (+ add & norm i)
// ff = feedforward (+ add & norm a)
struct Encoder<const O: usize> {
    a: Attention<O>,
    ff: FeedForward<O>,
}

struct Decoder {}
