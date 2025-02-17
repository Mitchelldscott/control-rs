//! # State-Space
//!
//! "The idea of **state-space** comes from the state-variable method of describing differential
//! equations. In this method, the differential equations describing the a dynamic system are
//! organized as a set of first order differential equations in the the vector-valued state of
//! the system, and the solution is visualized as a trajectory of this state vector in space.
//! **state-space control design** is the technique in which the control engineer designs a
//! dynamic compensation by working directly with the state-variable description of the system"
//! - 'Feedback Control of Dynamic Systems' by Gene F. Franklin, J. David Powell and Abbas
//! Emami-Naeini (ch 7.1)
//!
use nalgebra::{SMatrix, Scalar};
use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::DynamicModel;

/// Generic state-space model for a dynamic system
///
/// The state-space model represents a linear system of equations in the form:
///
/// <pre>
/// ẋ = A * x + B * u
/// y = C * x + D * u
/// </pre>
///
/// where:
/// - `x`: state vector (dimension N)
/// - `u`: input vector (dimension M)
/// - `y`: output vector (dimension L)
/// - `A`: system matrix (N x N)
/// - `B`: input matrix (N x M)
/// - `C`: output matrix (L x N)
/// - `D`: direct transmission matrix (L x M)
///
/// The model can describe single-input single-output (SISO), multiple-input multiple-output (MIMO),
/// and variations with differing numbers of states, inputs, and outputs.
///
/// # Generic Arguments
///
/// * `T` - scalar type of the system (e.g., `f32`, `f64`)
/// * `N` - number of state variables (dimension of `A`)
/// * `M` - number of input variables (dimension of `B`)
/// * `L` - number of output variables (dimension of `C`)
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix2, Matrix2x1, Matrix1x2, Matrix1};
/// use control_rs::StateSpace;
///
/// fn main() {
///     // Define the state-space matrices
///     let a = Matrix2::new(0.0, 1.0,
///                          -1.0, -0.1);
///     let b = Matrix2x1::new(0.0, 1.0);
///     let c = Matrix1x2::new(1.0, 0.0);
///     let d = Matrix1::new(0.0);
///
///     // Create the StateSpace model
///     let ss = StateSpace {
///         a,
///         b,
///         c,
///         d,
///     };
///
///     println!("{ss}");
/// }
/// ```
pub struct StateSpace<T, const N: usize, const M: usize, const L: usize> {
    /// system matrix `A` (N x N), representing the relationship between state derivatives and current states
    pub a: SMatrix<T, N, N>,
    /// input matrix `B` (N x M), representing how inputs affect state derivatives
    pub b: SMatrix<T, N, M>,
    /// output matrix `C` (L x N), representing how states contribute to outputs
    pub c: SMatrix<T, L, N>,
    /// direct transmission matrix `D` (L x M), representing direct input-to-output relationships
    pub d: SMatrix<T, L, M>,
}

impl<T, const N: usize, const M: usize, const L: usize> StateSpace<T, N, M, L>
where
    T: 'static + Copy + PartialEq + std::fmt::Debug,
{
    /// create a new state space model from lists of rows
    ///
    /// # Arguments
    ///
    /// * `a` - rows of the system matrix
    /// * `b` - rows of the input matrix
    /// * `c` - rows of the output matrix
    /// * `d` - rows of the direct transmission matrix
    ///
    /// # Returns
    ///
    /// * `StateSpace` - the generated state space model
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{Matrix2, Matrix2x1, Matrix1x2};
    /// use control_rs::{TransferFunction, StateSpace};
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
    ///     println!("{ss}");
    /// }
    /// ```
    pub fn new(a: [[T; N]; N], b: [[T; M]; N], c: [[T; N]; L], d: [[T; M]; L]) -> Self {
        StateSpace {
            a: SMatrix::from_fn(|i, j| a[i][j]),
            b: SMatrix::from_fn(|i, j| b[i][j]),
            c: SMatrix::from_fn(|i, j| c[i][j]),
            d: SMatrix::from_fn(|i, j| d[i][j]),
        }
    }
}

impl<T, Input, State, Output, const N: usize, const M: usize, const L: usize>
    DynamicModel<T, Input, State, Output> for StateSpace<T, N, M, L>
where
    T: 'static
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Copy
        + PartialEq
        + std::fmt::Debug,
    Input: Copy,
    State: Copy + Add<Output = State>,
    Output: Copy + Add<Output = Output>,
    SMatrix<T, N, N>: Mul<State, Output = State>,
    SMatrix<T, N, M>: Mul<Input, Output = State>,
    SMatrix<T, L, N>: Mul<State, Output = Output>,
    SMatrix<T, L, M>: Mul<Input, Output = Output>,
{
    fn f(&self, x: State, u: Input) -> State {
        self.a * x + self.b * u
    }

    fn h(&self, x: State, u: Input) -> Output {
        self.c * x + self.d * u
    }
}

impl<T, const N: usize, const M: usize, const L: usize> std::fmt::Display for StateSpace<T, N, M, L>
where
    T: 'static + Copy + PartialEq + std::fmt::Debug + std::fmt::Display,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "StateSpace:\nA{:}B{:}C{:}D{:}",
            self.a, self.b, self.c, self.d
        )
    }
}

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
) -> StateSpace<T, N, 1, 1>
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
pub fn zoh<T, const N: usize, const M: usize, const L: usize>(
    ss: &StateSpace<T, N, M, L>,
    ts: T,
) -> StateSpace<T, N, M, L>
where
    T: Copy + Scalar + Zero + One + From<u8>,
    SMatrix<T, N, N>: Add<Output = SMatrix<T, N, N>>
        + Mul<T, Output = SMatrix<T, N, N>>
        + Mul<SMatrix<T, N, N>, Output = SMatrix<T, N, N>>
        + Mul<SMatrix<T, N, M>, Output = SMatrix<T, N, M>>
        + Div<T, Output = SMatrix<T, N, N>>,
    SMatrix<T, N, M>: Mul<T, Output = SMatrix<T, N, M>>,
{
    let k: u8 = 10;
    let identity = SMatrix::<T, N, N>::identity();
    let psi = (0..k - 1).fold(identity, |psi, i| {
        identity + ss.a * ts * psi / T::from(k - i)
    });
    StateSpace {
        a: identity + ss.a * ts * psi,
        b: psi * ss.b * ts,
        c: ss.c,
        d: ss.d,
    }
}

#[cfg(test)]
mod basic_ss_tests {
    // not as productive as it could be...
    use super::*;
    use crate::transfer_function::{as_monic, TransferFunction};

    #[test]
    fn initialize_velocity_statespace() {
        let ss = StateSpace::new(
            [[0.0, 1.0], [0.0, -0.1]],
            [[0.0], [1.0]],
            [[1.0, 0.0]],
            [[0.0]],
        );

        assert_eq!(
            ss.a,
            nalgebra::Matrix2::new(0.0, 1.0, 0.0, -0.1),
            "System matrix incorrect"
        );
        assert_eq!(
            ss.b,
            nalgebra::Matrix2x1::new(0.0, 1.0),
            "Input matrix incorrect"
        );
        assert_eq!(
            ss.c,
            nalgebra::Matrix1x2::new(1.0, 0.0),
            "Output matrix incorrect"
        );
    }

    #[test]
    fn velocity_model_zoh_and_stability() {
        let ss = StateSpace::new(
            [[0.0, 1.0], [0.0, -0.1]],
            [[0.0], [1.0]],
            [[1.0, 0.0]],
            [[0.0]],
        );

        let ssd = zoh(&ss, 0.1 as f32);

        assert_eq!(
            ssd.a,
            nalgebra::Matrix2::new(1.0, 0.09950166, 0.0, 0.99004984),
            "System matrix incorrect"
        );

        // check if the eigen values are marginally stable
        match ssd.a.eigenvalues() {
            Some(eigens) => {
                assert!(
                    eigens[0].abs() <= 1.0,
                    "unstable eigen value ({}) in F {:}",
                    eigens[0],
                    ssd.a
                );
                assert!(
                    eigens[1].abs() < 1.0,
                    "unstable eigen value ({}) in F {:}",
                    eigens[0],
                    ssd.a
                );
            }
            None => panic!("discrete state-space model matrix does not have eigen values"),
        };
    }

    #[test]
    fn control_cannonical_test() {
        let tf = TransferFunction::new([2.0, 4.0], [1.0, 1.0, 4.0, 0.0, 0.0]);
        let (num, den) = as_monic(&tf);

        assert_eq!(
            den[0], 1.0,
            "Transfer Function denominator is not monic\n{tf}"
        );

        let ss: StateSpace<_, 4, 1, 1> = control_canonical(num, den);

        assert_eq!(
            ss.a,
            nalgebra::Matrix4::from_row_slice(&[
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, -1.0
            ]),
            "System matrix incorrect"
        );
        assert_eq!(
            ss.b,
            nalgebra::Matrix4x1::new(0.0, 0.0, 0.0, 1.0),
            "Input matrix incorrect"
        );
        assert_eq!(
            ss.c,
            nalgebra::Matrix1x4::new(4.0, 2.0, 0.0, 0.0),
            "Output matrix incorrect"
        );
    }
}
