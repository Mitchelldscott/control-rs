//! implementation of lead and lag compensators.
//!
//!
//! # Reference
//! * Feedback Control of Dynamic Systems Ch. 6.7.2
use control_rs::TransferFunction;
use super::Scalar;

use num_traits::One;

pub type LeadCompensator = TransferFunction<Scalar, 2, 2>;

pub fn lead_compensator(w_td: Scalar, phase_lead: Scalar) -> LeadCompensator {
    // 1. Convert phase lead from degrees to radians
    let phi_max_radians = phase_lead.to_radians();

    // 2. Calculate alpha (α)
    // sin(phi_max) = (1 - alpha) / (1 + alpha)
    // (1 + alpha) * sin(phi_max) = 1 - alpha
    // sin(phi_max) + alpha * sin(phi_max) = 1 - alpha
    // alpha * sin(phi_max) + alpha = 1 - sin(phi_max)
    // alpha * (sin(phi_max) + 1) = 1 - sin(phi_max)
    // alpha = (1 - sin(phi_max)) / (1 + sin(phi_max))
    let sin_phi_max = phi_max_radians.sin();
    let alpha = (Scalar::one() - sin_phi_max) / (Scalar::one() + sin_phi_max);

    // 3. Calculate tau (τ)
    // omega_max = 1 / (tau * sqrt(alpha))
    // tau = 1 / (omega_max * sqrt(alpha))
    let tau = Scalar::one() / (w_td * alpha.sqrt());

    TransferFunction::new([tau, Scalar::one()], [alpha * tau, Scalar::one()])
}