pub mod attention;
pub mod dense;
pub mod transformer;
pub use dense::*;
use na::{DefaultAllocator, MatrixViewMut, Scalar};
use nalgebra::{self as na, vector, Const, Matrix, RawStorage, RawStorageMut, SMatrix, SVector};
use std::ops::{Add, Mul};

const CHARS: &str = "abcdefghijklmnopqrstuvxyz.!,-'1234567890";
const BLOCK: usize = 4; // Symbols in a block of context
const E: usize = CHARS.len(); // Embedding size
type Float = f32;
type Tokens = SMatrix<Float, E, BLOCK>;

fn main() {}
