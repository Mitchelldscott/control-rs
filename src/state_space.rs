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

#[cfg(feature = "std")]
use std::{
    fmt,
    ops::{Add, Div, Mul, Neg},
};

#[cfg(not(feature = "std"))]
use core::{
    fmt,
    ops::{Add, Div, Mul, Neg},
};

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
pub struct StateSpace<A, B, C, D> {
    /// system matrix `A` (N x N), representing the relationship between state derivatives and current states
    pub a: A,
    /// input matrix `B` (N x M), representing how inputs affect state derivatives
    pub b: B,
    /// output matrix `C` (L x N), representing how states contribute to outputs
    pub c: C,
    /// direct transmission matrix `D` (L x M), representing direct input-to-output relationships
    pub d: D,
}

impl<A, B, C, D> StateSpace<A, B, C, D> {
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
    pub fn new(a: A, b: B, c: C, d: D) -> Self {
        StateSpace { a, b, c, d }
        //     a: SMatrix::from_fn(|i, j| a[i][j]),
        //     b: SMatrix::from_fn(|i, j| b[i][j]),
        //     c: SMatrix::from_fn(|i, j| c[i][j]),
        //     d: SMatrix::from_fn(|i, j| d[i][j]),
        // }
    }
}

impl<Input, State, Output, A, B, C, D>
    DynamicModel<Input, State, Output> for StateSpace<A, B, C, D>
where
    Input: Clone,
    State: Clone + Add<Output = State>,
    Output: Clone + Add<Output = Output>,
    A: Clone + Mul<State, Output = State>,
    B: Clone + Mul<Input, Output = State>,
    C: Clone + Mul<State, Output = Output>,
    D: Clone + Mul<Input, Output = Output>,
{
    fn dynamics(&self, x: State, u: Input) -> State {
        self.a.clone() * x + self.b.clone() * u
    }

    fn output(&self, x: State, u: Input) -> Output {
        self.c.clone() * x + self.d.clone() * u
    }
}

impl<A, B, C, D> fmt::Display for StateSpace<A, B, C, D>
where
    A: fmt::Display,
    B: fmt::Display,
    C: fmt::Display,
    D: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

#[cfg(test)]
mod basic_ss_tests {
    // not as productive as it could be...
    use super::*;
    use crate::transfer_function::{as_monic, TransferFunction};

    #[test]
    fn initialize_velocity_statespace() {
        let ss = StateSpace::new(
            nalgebra::Matrix2::new(0.0, 1.0, 0.0, -0.1),
            nalgebra::Matrix2x1::new(0.0, 1.0),
            nalgebra::Matrix1x2::new(1.0, 0.0),
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
            nalgebra::Matrix2::new(0.0, 1.0, 0.0, -0.1),
            nalgebra::Vector2::new(0.0, 1.0),
            nalgebra::Matrix1x2::new(1.0, 0.0),
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
        let monic_tf = as_monic(&tf);
        let (num, den) = (monic_tf.numerator, monic_tf.denominator.reduce_order("s"));

        assert_eq!(
            den[0], 1.0,
            "Transfer Function denominator is not monic\n{tf}"
        );

        let ss = control_canonical::<f64, 4, 2, 4>(
            num.coefficients().try_into().unwrap(),
            den.coefficients().try_into().unwrap(),
        );

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
