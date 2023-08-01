//!     "Attention Is All You Need" - This is the original research paper that introduced the transformer architecture, which has since become a cornerstone of NLP research. The paper provides a thorough explanation of the attention mechanism and its implementation in the transformer network.
//!
//!    "The Illustrated Transformer" - This is a popular blog post that provides a visual and intuitive explanation of the transformer architecture, including the attention mechanism.
//!
//!    "Annotated Transformer" - This is a resource that provides a step-by-step implementation of the transformer architecture in PyTorch, along with detailed explanations of each component.
//!
//!    Coursera's "Sequence Models" course - This is a deep learning course offered by deeplearning.ai that covers the transformer architecture and attention mechanism in depth, among other topics.
//!
//!    Fast.ai's Practical Deep Learning for Coders - This is a course that covers various deep learning techniques, including the transformer architecture and attention mechanism, in a practical and hands-on manner.

use crate::*;

pub struct Attention<const O: usize> {
    w_k: SMatrix<Float, O, E>,
    w_q: SMatrix<Float, O, E>,
    w_v: SMatrix<Float, O, E>,
    lr: Float,
    wd: Float, // TODO: impl weight decay
}

pub struct AttentionOutput<const O: usize> {
    qk: SMatrix<Float, O, O>,
    v: SMatrix<Float, O, BLOCK>,
    q: SMatrix<Float, O, BLOCK>,
    k: SMatrix<Float, O, BLOCK>,
    a: SMatrix<Float, O, BLOCK>,
}

pub struct AttentionInput {
    k: Tokens,
    q: Tokens,
    v: Tokens,
}

impl<const O: usize> Attention<O> {
    pub fn new(lr: Float, wd: Float) -> Self {
        Self {
            w_k: SMatrix::new_random() / O as Float,
            w_q: SMatrix::new_random() / O as Float,
            w_v: SMatrix::new_random() / O as Float,
            lr,
            wd,
        }
    }

    pub fn self_attention(&self, seq: Tokens) -> AttentionInput {
        AttentionInput {
            k: seq,
            q: seq,
            v: seq,
        }
    }

    pub fn attention(&self, i: &AttentionInput) -> AttentionOutput<O> {
        let AttentionInput { k, q, v } = i;
        let k = self.w_k * k;
        let q = self.w_q * q;
        let v = self.w_v * v;

        // in the paper q*k^T is divided by sqrt(d_k)
        let mut qk = (q * k.transpose()) / (CHARS.len() as Float).sqrt();
        for row in qk.row_iter_mut() {
            softmax(row.transpose())
        }
        let a = qk * v;

        AttentionOutput { k, q, v, qk, a }
    }

    pub fn backprop(
        &mut self,
        gradient: SMatrix<Float, O, BLOCK>,
        input: &AttentionInput,
        out: &AttentionOutput<O>,
    ) -> (Tokens, Tokens, Tokens) {
        let AttentionInput { k, q, v } = input;

        let dwv: SMatrix<f32, O, BLOCK> = out.qk.transpose() * gradient;

        let mut dsm = out.qk;
        for (row, d) in dsm
            .row_iter_mut()
            .zip((gradient * out.v.transpose()).row_iter())
        {
            softmax_backwards(row.transpose(), d.transpose());
        }

        // V should be part of this derivative.
        // Also sqrt(d_k) should be included somewhere.
        let sq_dk = (CHARS.len() as Float).sqrt();
        // q * kT
        // q' = dsm * k
        // kT' = qT * dsm
        let dwq: SMatrix<f32, O, BLOCK> = dsm * (out.k / sq_dk);
        let dwk: SMatrix<f32, O, BLOCK> = ((out.q.transpose() / sq_dk) * dsm).transpose();

        let dv = linear_backprop(v, &dwv, &mut self.w_v, self.lr);
        let dq = linear_backprop(q, &dwq, &mut self.w_q, self.lr);
        let dk = linear_backprop(k, &dwk, &mut self.w_k, self.lr);

        (dk, dq, dv)
    }
}

#[test]
fn linear_fiddy() {
    let iterations = 1000;

    let mut w1 = SMatrix::<Float, 3, 2>::new_random();
    let mut w2 = SMatrix::<Float, 2, 3>::new_random();

    let mut fiddy = [0., 0.];
    for n in 0..iterations {
        let r = |i| (n % i) as Float / i as Float + 1.;

        let i = SVector::from_column_slice(&[r(2), r(3)]);
        let y = SVector::from_column_slice(&[r(3), r(4)]);

        let out = w2 * (w1 * i);
        if out.iter().any(|n| n.is_nan() || n.is_infinite()) {
            panic!("Exploding / vanishing gradient");
        }
        let cost = (out - y).iter().map(|n| n.powi(2)).sum::<Float>();
        fiddy[n * 2 / iterations] += cost;

        let d = (out - y) * 2.;
        let d = linear_backprop(&(w1 * i), &d, &mut w2, 0.001);
        linear_backprop(&i, &d, &mut w1, 0.001);

        println!("{cost:.3}");
    }
    assert!(fiddy[0] > fiddy[1]);
}

#[test]
fn self_attention_fiddy() {
    let iterations = 1000;
    let mut fiddy = [0., 0.];

    let mut a = Attention::<{ CHARS.len() }>::new(0.01, 0.01);

    for n in 0..iterations {
        let mut v: SMatrix<Float, E, BLOCK> = SMatrix::new_random();
        for (j, mut col) in v.column_iter_mut().enumerate() {
            col *= (j as Float).sin() + (n as Float).cos();
        }

        let attended = n % 10;
        let mut y = v;
        for (n, mut col) in y.column_iter_mut().enumerate() {
            if n != attended {
                col *= 0.
            }
            let mut signal: SVector<Float, E> =
                SVector::from_iterator((0..E).map(|n| (n as Float).sin()));
            softmax(signal.column_mut(0));

            col.iter_mut().zip(signal.iter()).for_each(|(n, s)| *n *= s);
        }

        let i = a.self_attention(v);
        let o = a.attention(&i);
        let delta = (o.a - y) * 2.;
        a.backprop(delta, &i, &o);

        let mut cost = 0.;
        for r in 0..E {
            for c in 0..BLOCK {
                cost += (o.a[(r, c)] - y[(r, c)]).powi(2)
            }
        }
        cost /= (E * BLOCK) as Float;
        if cost.is_nan() || cost.is_infinite() {
            panic!("Exploding/ vanishing gradient");
        }
        fiddy[n * 2 / iterations] += cost;

        println!("cost: {cost}");
    }
    assert!(fiddy[0] > fiddy[1]);
}
