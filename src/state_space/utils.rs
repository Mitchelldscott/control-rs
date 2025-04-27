//! Utilities for creating, converting and analyzing SS models
//! 

use super::*;

/// create a new SISO state-space model from the numerator and denomenator coefficients of
/// a monic transfer function
///
/// The transfer function should be proper and its denominator must be monic. The function still
/// runs if this criteria is not met but the results will likely be incorrect.
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
/// where j = 0 is the constant and j = N - 1 is the second highest order coefficient
/// - the derivitive of each state variable i is the state variable i + 1
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
///
/// * `b` - coefficients of a transfer functions numerator `[b_n ... b_1]`
/// * `a` - coefficients of a transfer functions denominator `[1.0, a_n .. a_1]`
///
/// # Returns
///
/// * `StateSpace` - state-space model in control canonical form
///
/// # Example
///
/// ```
/// use control_rs::state_space::{StateSpace, control_canonical};
///
/// fn main() {
///     let ss: StateSpace::<_,2,1,1> = control_canonical([1.0], [0.1, 0.0]);
///     println!("{ss}");
/// }
/// ```
pub fn control_canonical<T, const N: usize, const M: usize, const L: usize>(
    b: [T; M],
    a: [T; L],
) -> StateSpace<SMatrix<T,N,N>, SMatrix<T,N,1>, SMatrix<T,1,N>, SMatrix<T,1,1>>
where
    T: 'static + Copy + Scalar + Zero + One + Neg<Output = T>,
{
    StateSpace {
        a: SMatrix::from_fn(|i, j| {
            if i == N - 1 {
                -a[L - j - 1]
            } else {
                if i + 1 == j {
                    T::one()
                } else {
                    T::zero()
                }
            }
        }),
        b: SMatrix::from_fn(|i, _| if i == N - 1 { T::one() } else { T::zero() }),
        c: SMatrix::from_fn(|_, j| if j < M { b[M - j - 1] } else { T::zero() }),
        d: SMatrix::from_fn(|_, _| T::zero()),
    }
}

/// Discretizes the given StateSpace model
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
/// use control_rs::state_space::{StateSpace, zoh};
///
/// fn main() {
///     let ss = StateSpace::new(
///         [
///             [0.0, 1.0],
///             [0.0, -0.1]
///         ],
///         [
///             [0.0],
///             [1.0]
///         ],
///         [
///             [1.0, 0.0]
///         ],
///         [[0.0]]
///     );
///
///     let ssd = zoh(&ss, 0.1 as f32);
///     println!("{ssd}");
/// }
/// ```
///
/// ## TODO:
/// - [ ] compare with [nalgebra::Matrix::exp]
pub fn zoh<T, A, B, C, D>(
    ss: &StateSpace<A, B, C, D>,
    ts: T,
) -> StateSpace<A, B, C, D>
where
    T: Copy + Scalar + Zero + One + From<u8>,
    A: Clone 
        + Add<Output = A>
        + Mul<T, Output = A>
        + Div<T, Output = A>
        + Mul<A, Output = A>
        + Mul<B, Output = B>
        + Default,
    B: Clone
        + Mul<T, Output = B>,
    C: Clone,
    D: Clone,
{
    let k: u8 = 10;
    let identity = A::default(); // need an identity trait?
    let psi = (0..k - 1).fold(identity.clone(), |psi, i| {
        identity.clone() + ss.a.clone() * ts * psi.clone() / T::from(k - i)
    });
    StateSpace {
        a: identity + ss.a.clone() * ts * psi.clone(),
        b: psi * ss.b.clone() * ts,
        c: ss.c.clone(),
        d: ss.d.clone(),
    }
}