//! # State-Space
//!
//! "The idea of **state-space** comes from the state-variable method of describing differential
//! equations. In this method, the differential equations describing a dynamic system are
//! organized as a set of first order differential equations in the vector-valued state of
//! the system. The solution is visualized as a trajectory of this state vector in space.
//! **state-space control design** is the technique in which the control engineer designs a
//! dynamic compensation by working directly with the state-variable description of the system"
//! - 'Feedback Control of Dynamic Systems' by Gene F. Franklin, J. David Powell and Abbas
//!   Emami-Naeini (ch 7.1)
//!

use core::{
    fmt,
    ops::{Add, Mul},
};

use crate::math::systems::DynamicalSystem;

// ===============================================================================================
//      StateSpace Tests
// ===============================================================================================

#[cfg(test)]
mod basic_ss_tests;

// ===============================================================================================
//      StateSpace Sub-modules
// ===============================================================================================

pub mod utils;
pub use utils::*;

// ===============================================================================================
//      StateSpace Model
// ===============================================================================================

/// Generic state-space model for a dynamic system
///
/// The state-space model represents a linear system of equations in the form:
///
/// <pre>
/// ẋ = A * x + B * u
/// y = C * x + D * u
/// </pre>
///
/// Where:
/// - `x`: state vector (dimension N)
/// - `u`: input vector (dimension M)
/// - `y`: output vector (dimension L)
/// - `A`: system matrix (N x N)
/// - `B`: input matrix (N x M)
/// - `C`: output matrix (L x N)
/// - `D`: direct transmission matrix (L x M)
///
/// The model can describe single-input single-output (SISO), multiple-input multiple-output (MIMO),
/// single-input multiple-output (SIMO) and multiple-input single-output (MISO).
///
/// # Generic Arguments
/// * `A` - system matrix type
/// * `B` - input matrix type
/// * `C` - input matrix type
/// * `D` - direct transmission matrix type
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix2, Matrix2x1, Matrix1x2, Matrix1};
/// use control_rs::StateSpace;
/// // Define the state-space matrices
/// let a = Matrix2::new(0.0, 1.0, -1.0, -0.1);
/// let b = Matrix2x1::new(0.0, 1.0);
/// let c = Matrix1x2::new(1.0, 0.0);
/// let d = Matrix1::new(0.0);
/// // Create the StateSpace model
/// let ss = StateSpace { a, b, c, d, };
/// println!("{ss}");
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
    /// Create a new state space model from lists of rows
    ///
    /// # Arguments
    /// * `a` - rows of the system matrix
    /// * `b` - rows of the input matrix
    /// * `c` - rows of the output matrix
    /// * `d` - rows of the direct transmission matrix
    ///
    /// # Returns
    /// * `StateSpace` - the generated state space model
    ///
    /// # Example
    ///
    /// ```
    /// use control_rs::StateSpace;
    /// let ss = StateSpace::new(
    ///      [[0.0, 1.0], [0.0, -0.1]],
    ///      [[0.0], [1.0]],
    ///      [[1.0, 0.0]],
    ///      [[0.0]]
    ///  );
    ///  println!("{ss:?}");
    /// ```
    pub const fn new(a: A, b: B, c: C, d: D) -> Self {
        Self { a, b, c, d }
    }
}

// ===============================================================================================
//      StateSpace as DynamicModel
// ===============================================================================================

impl<Input, State, Output, A, B, C, D> DynamicalSystem<Input, State, Output>
    for StateSpace<A, B, C, D>
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

// ===============================================================================================
//      StateSpace Format
// ===============================================================================================

impl<A, B, C, D> fmt::Display for StateSpace<A, B, C, D>
where
    A: fmt::Display,
    B: fmt::Display,
    C: fmt::Display,
    D: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StateSpace:\nA{:}B{:}C{:}D{:}",
            self.a, self.b, self.c, self.d
        )
    }
}

impl<A, B, C, D> fmt::Debug for StateSpace<A, B, C, D>
where
    A: fmt::Debug,
    B: fmt::Debug,
    C: fmt::Debug,
    D: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StateSpace:\nA{:?}B{:?}C{:?}D{:?}",
            self.a, self.b, self.c, self.d
        )
    }
}
