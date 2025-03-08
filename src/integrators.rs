//! A variety of methods to integrate [DynamicSystem]s
//! 
use super::DynamicModel;

#[cfg(feature = "std")]
use std::ops::{Add, Div, Mul};

#[cfg(not(feature = "std"))]
use core::ops::{Add, Div, Mul};

/// Integrate the system for a given time interval
///
/// The time interval is assumed to be small enough the input will be constant for the
/// duration of the integration. For simulations with time-varying input call this repeatedly
/// in a loop.
///
/// # Arguments
///
/// * `x0` - initial state
/// * `u` - input
/// * `t0` - start time
/// * `tf` - end time
/// * `dt` - length of a step
///
/// # Returns
///
/// * `x` - state at end time
pub fn runge_kutta4<T, X, U, Y, Sys>(model: &Sys, x0: X, u: U, t0: T, tf: T, dt: T) -> X 
where 
    Y: Copy,
    U: Copy,
    X: Copy + Add<Output = X> + Mul<T, Output = X>,
    T: Copy + PartialOrd + Add<Output = T> + Div<Output = T> + From<u8>,
    Sys: DynamicModel<U, X, Y>,
{
    let dt_2 = dt / T::from(2);
    let dt_3 = dt / T::from(3);
    let dt_6 = dt / T::from(6);

    let mut t = t0;
    let mut x = x0;

    while t < tf {
        let k1 = model.dynamics(x, u);
        let k2 = model.dynamics(x + k1 * dt_2, u);
        let k3 = model.dynamics(x + k2 * dt_2, u);
        let k4 = model.dynamics(x + k3 * dt_2, u);
        x = x + k1 * dt_6 + k2 * dt_3 + k3 * dt_3 + k4 * dt_6;
        t = t + dt;
    }
    x
}