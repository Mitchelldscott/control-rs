//! Quadcopter attitude and altitude controls
//! TODO:
//!     * Fix NLDynamics
//!         * fix bugs
//!         * move to separate mod like dc_motor
//!         * double check ground implementation
//!     * Change linearization to use analytical derivation of jacobians not perturbations
//!     * Find paper with good control architecture
//!     * Implement controller and compare performance with paper

use control_rs::{
    StateSpace,
    integrators::runge_kutta4,
    systems::{DynamicalSystem, NLModel},
};
use nalgebra::{Matrix3, Rotation3, SMatrix, SVector, Vector3, Vector4};

/// Calculates the Direction Cosine Matrix (DCM) from Euler angles (roll, pitch, yaw).
///
/// This function is equivalent to the Z-Y-X rotation sequence Rz(yaw) * Ry(pitch) * Rx(roll).
///
/// # Arguments
///
/// * `q` - A 3D vector containing the Euler angles `[roll, pitch, yaw]` in radians.
pub fn dcm(theta: f64, phi: f64, psi: f64) -> Rotation3<f64> {
    // nalgebra's `from_euler_angles` creates a rotation matrix using
    // the Z-Y'-X'' sequence, which is equivalent to the Python code's
    // Rz @ Ry @ Rx multiplication.
    // q.x is roll, q.y is pitch, q.z is yaw.
    Rotation3::from_euler_angles(theta, phi, psi)
}

/// Creates a skew-symmetric matrix from a 3D vector.
///
/// This matrix represents the cross-product operation, such that
/// `tilde(v) * w` is equivalent to `v.cross(w)`.
///
/// # Arguments
///
/// * `v` - A 3D vector.
pub fn tilde(v: &Vector3<f64>) -> Matrix3<f64> {
    // nalgebra provides a direct method to create this matrix.
    v.cross_matrix()
}

/// Holds the nonlinear dynamics parameters and state for a quadcopter.
#[derive(Debug)]
pub struct QuadcopterNLDynamics {
    pub dt: f64,
    pub prev_u: Vector4<f64>,
    pub delta_u: Vector4<f64>,

    // Parameters
    pub gravity: f64,
    pub thrust_coefficient: f64,
    pub wing_length: f64,
    pub prop_length: f64,
    pub airframe_mass: f64,
    pub prop_drag_coefficient: f64,
    pub frame_drag_coefficient: f64,
    pub prop_inertia: f64,
    pub inertia: Matrix3<f64>,

    // Weighting matrices (e.g., for control)
    pub qa: Matrix3<f64>,
    pub qw: Matrix3<f64>,
    pub ra: Matrix3<f64>,
    pub rw: Matrix3<f64>,
}

impl Default for QuadcopterNLDynamics {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadcopterNLDynamics {
    /// Creates a new `QuadcopterNLDynamics` instance with default parameters.
    pub fn new() -> Self {
        Self {
            dt: 0.001,
            prev_u: Vector4::zeros(),
            delta_u: Vector4::zeros(),

            // Parameters
            gravity: 9.81,
            thrust_coefficient: 0.02,
            wing_length: 0.23,
            prop_length: 0.25,
            airframe_mass: 0.025,
            prop_drag_coefficient: 0.0001,
            frame_drag_coefficient: 0.001,
            prop_inertia: 0.00017,

            // The inertia matrix is constructed from its diagonal elements and scaled.
            inertia: Matrix3::from_diagonal(&Vector3::new(3.2132169, 5.0362409, 7.22155076)) * 1e-3,

            // Weighting matrices are initialized as identity matrices.
            qa: Matrix3::identity(),
            qw: Matrix3::identity(),
            ra: Matrix3::identity(),
            rw: Matrix3::identity(),
        }
    }
}

type QuadInput = SVector<f64, 4>;
type QuadState = SVector<f64, 12>;
type QuadOutput = SVector<f64, 12>; // full state output for now

impl DynamicalSystem<QuadInput, QuadState, QuadOutput> for QuadcopterNLDynamics {
    /// Implements the nonlinear quadcopter equations of motion: x_dot = f(x, u).
    fn dynamics(&self, x: QuadState, u: QuadInput) -> QuadState {
        // Deconstruct state vector for clarity
        let position = x.fixed_rows::<3>(0);
        let velocity = x.fixed_rows::<3>(3);
        let angles = x.fixed_rows::<3>(6); // [roll, pitch, yaw]
        let ang_velocity = x.fixed_rows::<3>(9); // [p, q, r]

        // --- Kinematics ---
        // 1. Position derivative (dx/dt = v)
        let pos_dot = if position[2] >= 0.0 {
            Vector3::new(velocity[0], velocity[1], velocity[2])
        } else {
            Vector3::new(velocity[0], velocity[1], 0.0)
        };

        // 2. Euler angle rates (d(angles)/dt = W * w)
        let (phi, theta, psi) = (angles[0], angles[1], angles[2]);
        let s_phi = phi.sin();
        let c_phi = phi.cos();
        let c_th = theta.cos();
        let t_th = theta.tan();

        // Transformation matrix from body rates to Euler rates
        let mut w_matrix = Matrix3::zeros();
        if c_th.abs() > 1e-6 {
            // Avoid singularity at pitch = +/- 90 deg
            w_matrix.m11 = 1.0;
            w_matrix.m12 = s_phi * t_th;
            w_matrix.m13 = c_phi * t_th;
            w_matrix.m22 = c_phi;
            w_matrix.m23 = -s_phi;
            w_matrix.m32 = s_phi / c_th;
            w_matrix.m33 = c_phi / c_th;
        }
        let angles_dot = w_matrix * ang_velocity;

        // --- Dynamics ---
        // Total thrust
        let total_thrust = self.thrust_coefficient * u.map(|u_i| u_i.powi(2)).sum();

        // Rotation matrix from body to inertial frame
        let r_matrix = dcm(theta, phi, psi);

        // 3. Acceleration in inertial frame (dv/dt = g - R*T/m)
        let thrust_inertial = r_matrix * Vector3::new(0.0, 0.0, total_thrust);
        let gravity_vec = Vector3::new(0.0, 0.0, -self.gravity);
        let vel_dot = if position[2] >= 0.0 {
            gravity_vec + (thrust_inertial / self.airframe_mass)
        } else {
            thrust_inertial / self.airframe_mass
        };
        // 4. Angular acceleration (dw/dt = J^-1 * (tau - w x Jw))
        let l = self.wing_length;
        let k = self.thrust_coefficient;
        let cd = self.prop_drag_coefficient;

        // Torques in body frame
        // TODO: add angular momentum of props to yaw torque
        let tau = Vector3::new(
            l * k * (u[3].powi(2) - u[1].powi(2)), // Roll torque
            l * k * (u[2].powi(2) - u[0].powi(2)), // Pitch torque
            cd * (-u[0].powi(2) + u[1].powi(2) - u[2].powi(2) + u[3].powi(2)), // Yaw torque
        );

        let j = &self.inertia;
        let j_inv = j.try_inverse().unwrap_or_else(Matrix3::identity);
        let gyro_effect = ang_velocity.cross(&(j * ang_velocity));
        let ang_vel_dot = j_inv * (tau - gyro_effect);

        // Reconstruct the state derivative vector
        let mut x_dot = QuadState::zeros();
        x_dot.fixed_rows_mut::<3>(0).copy_from(&pos_dot);
        x_dot.fixed_rows_mut::<3>(3).copy_from(&vel_dot);
        x_dot.fixed_rows_mut::<3>(6).copy_from(&angles_dot);
        x_dot.fixed_rows_mut::<3>(9).copy_from(&ang_vel_dot);

        x_dot
    }

    /// The output is the full state vector, so y = h(x,u) = x.
    fn output(&self, x: QuadState, _u: QuadInput) -> QuadOutput {
        x
    }
}

type QuadcopterSS = StateSpace<
    SMatrix<f64, 12, 12>,
    SMatrix<f64, 12, 4>,
    SMatrix<f64, 12, 12>,
    SMatrix<f64, 12, 4>,
>;

/// This implementation allows the quadcopter model to be linearized at any
/// operating point (x, u).
impl
    NLModel<
        QuadInput,
        QuadState,
        QuadOutput,
        SMatrix<f64, 12, 12>,
        SMatrix<f64, 12, 4>,
        SMatrix<f64, 12, 12>,
        SMatrix<f64, 12, 4>,
    > for QuadcopterNLDynamics
{
    /// Linearizes the system about a nominal state and input using the
    /// finite difference method to approximate the Jacobians.
    fn linearize(&self, x: QuadState, u: QuadInput) -> QuadcopterSS {
        let mut a = SMatrix::<f64, 12, 12>::zeros();
        let mut b = SMatrix::<f64, 12, 4>::zeros();
        let c = SMatrix::<f64, 12, 12>::identity(); // Since y = x, C is the identity matrix
        let d = SMatrix::<f64, 12, 4>::zeros(); // Since y does not depend on u, D is zero

        // Epsilon for numerical differentiation
        let eps = 1e-6;

        // --- Calculate A Matrix (Jacobian df/dx) ---
        let fx_nominal = self.dynamics(x, u);
        for i in 0..12 {
            let mut x_perturbed = x;
            x_perturbed[i] += eps;
            let fx_perturbed = self.dynamics(x_perturbed, u);
            let column = (fx_perturbed - fx_nominal) / eps;
            a.set_column(i, &column);
        }

        // --- Calculate B Matrix (Jacobian df/du) ---
        for i in 0..4 {
            let mut u_perturbed = u;
            u_perturbed[i] += eps;
            let fx_perturbed = self.dynamics(x, u_perturbed);
            let column = (fx_perturbed - fx_nominal) / eps;
            b.set_column(i, &column);
        }

        StateSpace { a, b, c, d }
    }
}

// Example of how to use it
fn main() {
    // Initialize the dynamics
    let quad_dynamics = QuadcopterNLDynamics::new();
    let mut quad_state = QuadState::zeros();
    // F = ma = 4 * tau * U^2
    let grav_compensation = (quad_dynamics.airframe_mass * quad_dynamics.gravity
        / (4.0 * quad_dynamics.thrust_coefficient))
        .sqrt();
    let quad_input = QuadInput::new(
        grav_compensation + 0.01,
        grav_compensation - 0.01,
        grav_compensation + 0.01,
        grav_compensation - 0.01,
    );
    for _ in 0..1000 {
        quad_state = runge_kutta4(&quad_dynamics, quad_state, quad_input, 0.0, 0.1, 0.01);
    }

    println!("{:?}", quad_state);
}
