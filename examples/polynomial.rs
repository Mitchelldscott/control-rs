
use core::ops::{Add, AddAssign, Mul};
use num_traits::Zero;
use nalgebra::{Const, DimMax};

struct Polynomial<T, const N: usize> {
    /// the data
    pub coefficients: [T; N]
}

impl<T, const N: usize, const M: usize, const L: usize> Add<Polynomial<T, M>> for Polynomial<T, N>
where
    T: Copy + Zero + AddAssign,
    Const<N>: DimMax<Const<M>, Output = Const<L>>,
{
    type Output = Polynomial<T, L>;
    fn add(self, rhs: Polynomial<T, M>) -> Self::Output {
        let mut result = Polynomial { coefficients: [T::zero(); L] };
        for (i, c) in result.coefficients.iter_mut().enumerate() {
            // SAFETY: `i < N` is a valid index of [T; N]
            if i < N { unsafe { *c += self.coefficients.get_unchecked(i).clone() } }
            // SAFETY: `i < M` is a valid index of [T; M]
            if i < M { unsafe { *c += rhs.coefficients.get_unchecked(i).clone() } }
        }
        result
    }
}

fn main()  {
    let p1 = Polynomial { coefficients: [1i32; 0] };
    let p2 = Polynomial { coefficients: [0i32; 0] };
    
    assert_eq!((p1 + p2).coefficients, [1i32; 0], "wrong addition result")
}