use crate::*;

pub fn one_hot(s: &str) -> Tokens {
    assert_eq!(s.len(), BLOCK);
    SMatrix::from_iterator(s.chars().flat_map(|c| {
        let mut col = [0.; CHARS.len()];
        col[CHARS.find(c).unwrap()] = 1.;
        col
    }))
}

pub fn softmax<const N: usize, S: RawStorageMut<Float, Const<N>, Const<1>>>(
    mut v: Matrix<Float, Const<N>, Const<1>, S>,
) {
    let sum: Float = v.iter().map(|n| n.exp()).sum();
    for n in 0..N {
        v[n] = v[n].exp() / sum
    }
}

pub fn softmax_backwards<const N: usize, S: RawStorageMut<Float, Const<N>, Const<1>>>(
    mut v: Matrix<Float, Const<N>, Const<1>, S>,
    grad: Matrix<Float, Const<N>, Const<1>, S>,
) {
    let sum: Float = v.iter().map(|n| n.exp()).sum();
    for n in 0..N {
        // TODO: fix this
        v[n] = v[n] * (1. - v[n]) * grad[n]
    }
}

pub fn linear_backprop<const O: usize, const I: usize, const J: usize>(
    input: &SMatrix<f32, I, J>,
    gradient: &SMatrix<f32, O, J>,
    w: &mut SMatrix<f32, O, I>,
    lr: Float,
) -> SMatrix<f32, I, J> {
    // y = f(h(x))
    // h'(x) = w * x
    // dy/dw = f'(h(x)) * xT
    // dy/dx = wT * f'(h(x))
    *w -= gradient * input.transpose() * lr / O as Float;
    w.transpose() * gradient
}
