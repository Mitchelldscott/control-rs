//! Utilities for creating, converting and analyzing SS models
//!

use core::ops::{Add, Div, Mul, Neg, Sub};

use nalgebra::{Const, DimSub, SMatrix, Scalar, U1};
use num_traits::{One, Zero};

use super::StateSpace;

/// Create a new SISO state-space model from the numerator and denominator coefficients of
/// a monic transfer function
///
/// The transfer function should be proper and its denominator must be monic. The function still
/// runs if this criterion is not met, but the results will be incorrect.
///
/// <pre>
/// G(s) = b(s) / a(s)
/// a(s) = (s^N + a_N s^(N-1) + ... + a_1)
/// b(s) = (b_N s^(N-1) + b_(N-1) s^(N-2) + ... + b_1)
///
/// A = [  0    1    0  ...  0   ]
///     [  0    0    1  ...  0   ]
///     [  ...                   ]
///     [ -a_1 -a_2 -a_3... -a_n ]
///  
/// - the final row of A is the denominator coefficients
///   where j = 0 is the constant and j = N - 1 is the second-highest order coefficient
/// - the derivative of each state variable i is the state variable `i + 1`
///
/// B = [ 0 ]
///     [...]
///     [ 1 ]
///
/// - the input is a term in the derivative of x_n
///
/// C = [ b_1 b_2 ... b_n]
///
/// D = [ 0 ]
/// </pre>
///
/// # Arguments
/// * `b` - coefficients of a transfer function's numerator `[b_n ... b_1]`
/// * `a` - coefficients of a transfer function's denominator `[a_n ... a_1]`
///
/// # Returns
/// * `StateSpace` - state-space model in control canonical form
///
/// # Example
///
/// ```
/// use control_rs::state_space::{StateSpace, utils::control_canonical};
/// let ss = control_canonical(&[1.0], &[0.1, 0.0]);
/// println!("{ss}");
/// ```
/// # TODO: repair docs
pub fn control_canonical<T, const N: usize, const M: usize, const L: usize>(
    b: &[T; M],
    a: &[T; L],
) -> StateSpace<SMatrix<T, N, N>, SMatrix<T, N, 1>, SMatrix<T, 1, N>, SMatrix<T, 1, 1>>
where
    T: Scalar + Clone + Zero + One + Neg<Output = T> + Div<Output = T>,
    Const<L>: DimSub<U1, Output = Const<N>>,
{
    StateSpace {
        a: SMatrix::from_fn(|i, j| {
            if i == N - 1 {
                a[j].clone().neg() / a[L - 1].clone()
            } else if i + 1 == j {
                T::one()
            } else {
                T::zero()
            }
        }),
        b: SMatrix::from_fn(|i, _| if i == N - 1 { T::one() } else { T::zero() }),
        c: SMatrix::from_fn(|_, j| if j < M { b[j].clone() } else { T::zero() }),
        d: SMatrix::from_fn(|_, _| T::zero()),
    }
}

/// Discretizes the given `StateSpace` model
///
/// This discretization applies a zero-order-hold (zoh) to the continuous state space matrices.
/// zoh relies on the matrix exponent operation e^At which is approximated using a tenth order
/// taylor-series expansion. The implementation is based on an example from "Digital Control of
/// Dynamic Systems" by Gene F. Franklin, J. David Powell and Michael Workman (Ch 4.3, pg 107).
///
/// 1. Select sampling period T
/// 2. **I** = Identity
/// 3. **Psi** = Identity
/// 4. k = 10
/// 5. for i in 0..k-1
///     - **Psi** = **I** + **A** * T * **Psi** / (k - i)
/// 6. **G** = T * **Psi** * **B**
/// 7. **F** = **I** + **F** * T * **Psi**
///
/// # Arguments
///
/// * `&self` - the dynamic system to linearize
/// * `ts` - sampling time of the system
/// * `x` - state vector
/// * `u` - input vector
///
/// # Returns
///
/// * `StateSpace` - a discretized version of the state space model
///
/// # Example
///
/// ```
/// use control_rs::state_space::utils::{control_canonical, zoh};
/// let ss = control_canonical(&[1.0, 1.0], &[1.0, 1.0, 1.0]);
/// let ssd = zoh(&ss, 0.1_f32);
/// println!("{ssd}");
/// ```
///
/// ## TODO:
/// - [ ] compare with [`nalgebra::Matrix::exp`]
pub fn zoh<T, A, B, C, D>(ss: &StateSpace<A, B, C, D>, ts: T) -> StateSpace<A, B, C, D>
where
    T: Clone + Zero + One + Sub<Output = T>,
    A: Clone
        + One
        + Add<Output = A>
        + Mul<T, Output = A>
        + Div<T, Output = A>
        + Mul<A, Output = A>
        + Mul<B, Output = B>
        + Default,
    B: Clone + Mul<T, Output = B>,
    C: Clone,
    D: Clone,
{
    let k = 10;
    // TODO: Really need constants to remove this garbage
    let k_as_t = (0..k).fold(T::zero(), |t, _| T::one() + t.clone());
    let identity = A::one();
    // let psi = (0..k - 1).fold(identity.clone(), |psi, i| {
    //     identity.clone() + ss.a.clone() * ts.clone() * psi / T::from(k - i)
    // });
    let (_, psi) = (0..k - 1).fold((k_as_t, identity.clone()), |(k, psi), _| {
        (
            k.clone() - T::one(),
            identity.clone() + ss.a.clone() * ts.clone() * psi / k,
        )
    });
    StateSpace {
        a: identity + ss.a.clone() * ts.clone() * psi.clone(),
        b: psi * ss.b.clone() * ts,
        c: ss.c.clone(),
        d: ss.d.clone(),
    }
}
