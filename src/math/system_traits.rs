//! higher level traits for numerical models

/// # Numerical Function trait
///
/// This trait provides a universal interface for evaluating numerical models.
///
/// Model must be in the form:
/// <pre>
/// y = f(x)
/// </pre>
pub trait NumericalFunction<T> {
    /// Evaluates the function for the given input
    fn __evaluate(&self, x: T) -> T;
}

/// # Dynamic Model
///
/// This trait provides a universal interface for evalutating numerical models.
///
/// Model must be in the form:
/// <pre>
/// ẋ = f(x, u)
/// y = h(x, u)
/// </pre>
///
/// # Generic Arguments
///
/// * `Input` - type of the input variable(s)
/// * `State` - type of the state variable(s)
/// * `Output` - type of the output variable(s)
pub trait DynamicModel<Input, State, Output> {
    /// Evaluates the dynamics of the state for the given state and input
    fn dynamics(&self, x: State, u: Input) -> State;
    /// Evaluates the model's output for the given state and input
    fn output(&self, x: State, u: Input) -> Output;
}

/// # Nonlinear Model
///
/// This allows users to implement a linearization of a nonlinear model. This also provides a
/// trait bound for algorithms that use linearization.
///
/// # Generic Arguments
///
/// * `T` - type of the state, input and output values
/// * `Input` - type of the input vector
/// * `State` - type of the state vector
/// * `Output` - type of the output vector
///
/// ## References
///
/// - *Nonlinear Systems*, Khalil, Ch. 2: Nonlinear Models.
///
/// ## TODO:
/// - [ ] move generics to type aliases, the <> are too full
/// - [ ] add generic linearization so users don't need to define a custom one (derive?)
/// - [ ] add LinearModel trait so custom models can be linearized to other forms (linear multivariate polynomial?)
pub trait NLModel<Input, State, Output, A, B, C, D>: DynamicModel<Input, State, Output> {
    /// Linearizes the system about a nominal state and input
    fn linearize(&self, x: State, u: Input) -> crate::state_space::StateSpace<A, B, C, D>;
}