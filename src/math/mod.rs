//! Shared behavior for mathematical types

mod assert_f;

mod num_traits;

pub mod complex_number;
pub mod systems;

/// Default precision value for all model formatters
pub const DEFAULT_PRECISION: usize = 3;
