//! # Frequency Tools
//!
//! The `frequency_tools` module provides utilities for analyzing and visualizing the frequency
//! response of linear systems, such as control systems and signal processing filters. These
//! tools are designed to compute key metrics like crossover frequencies and stability margins,
//! as well as to generate data for plots such as Bode and Nyquist diagrams.
//!
//! ## Features
//!
//! - **Crossover Frequency Calculation:** Find the frequencies where the system's magnitude and
//!   phase cross specific thresholds (e.g., gain crossover at 0 dB, phase crossover at -180°).
//! - **Stability Margins:** Compute gain margins (dB) and phase margins (degrees) to assess
//!   system stability.
//! - **Magnitude and Phase Response:** Evaluate systems at logarithmically spaced
//!   frequency points to compute magnitude and phase.
//! - **Visualization Backend:** Generate html plots using plotly-rs.
//!
//! ## Applications
//!
//! - **Control System Design:** Use frequency response metrics to tune controllers and ensure
//!   desired stability margins.
//! - **Filter Analysis:** Examine the frequency characteristics of digital and analog filters.
//! - **Robustness Evaluation:** Assess how systems behave under varying conditions by analyzing
//!   frequency domain behavior.
//!
//! ## References
//!
//! - Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2015). *Feedback Control of Dynamic Systems*.
//! - Dorf, R. C., & Bishop, R. H. (2016). *Modern Control Systems*.
//!
//! ### TODO:
//! - [ ] remove all the unwraps
//!     - `first_crossover` should get broken up to return the index of the crossover
//!     - Need a frequency field trait to provide some constants and log fns
//! - [ ] textbook example of trait productivity
//! - [ ] move plotly to plotly helper file (or wait for a nice gui)

use core::ops::{AddAssign, Div, Mul, Sub};
#[cfg(feature="std")]
use core::ops::Neg;

use nalgebra::{Complex, ComplexField, RealField};
use num_traits::{Float, One, Zero};

use crate::static_storage::array_from_iterator;

/// A trait for types that can represent an angle and provide
/// conversion from degrees to radians.
pub trait Phase: Sized + Mul<Output = Self> + Div<Output = Self> {
    /// Associated const to help with generic functions
    const PI: Self;
    /// Associated const to help with generic functions
    const ONE_EIGHTY: Self;
    /// Converts the angle from degrees to radians.
    #[must_use]
    #[inline]
    fn to_radians(self) -> Self { self * (Self::PI / Self::ONE_EIGHTY) }
    /// Converts the angle from radians to degrees.
    #[must_use]
    #[inline]
    fn to_degrees(self) -> Self { self * (Self::ONE_EIGHTY / Self::PI) }
}

impl Phase for f32 {
    const PI: Self = core::f32::consts::PI;
    const ONE_EIGHTY: Self = 180.0;
}
impl Phase for f64 {
    const PI: Self = core::f64::consts::PI;
    const ONE_EIGHTY: Self = 180.0;
}

/// A trait for types that can represent a magnitude and provide
/// conversion between absolute values and decibels (dB).
pub trait Magnitude: Sized + Mul<Output = Self> + Div<Output = Self> {
    /// Associated constant for magnitude calculations.
    const TEN: Self;

    /// Associated constant for decibel calculations.
    const TWENTY: Self;

    /// Calculates x where 10^x = self
    ///
    /// This is not defined for values <= 0
    #[must_use]
    fn log10(self) -> Self;

    /// Calculates self^exp
    ///
    /// This is not defined for values <= 0
    #[must_use]
    fn powf(self, exp: Self) -> Self;

    /// Converts an absolute magnitude value to decibels (dB).
    ///
    /// The formula used is `20 * log10(self / DB_REFERENCE)`.
    #[must_use]
    #[inline]
    fn to_db(self) -> Self { Self::TWENTY * self.log10() }

    /// Converts a decibel (dB) value to an absolute magnitude.
    ///
    /// The formula used is `DB_REFERENCE * 10^(self / DB_MULTIPLIER)`.
    #[must_use]
    #[inline]
    fn to_mag(self) -> Self { Self::TEN.powf(self / Self::TWENTY) }
}

impl Magnitude for f32 {
    const TEN: Self = 10.0;
    const TWENTY: Self = 20.0;
    #[inline]
    fn log10(self) -> Self { self.log10() }
    #[inline]
    fn powf(self, exp: Self) -> Self { self.powf(exp) }
}

impl Magnitude for f64 {
    const TEN: Self = 10.0;
    const TWENTY: Self = 20.0;
    #[inline]
    fn log10(self) -> Self { self.log10() }
    #[inline]
    fn powf(self, exp: Self) -> Self { self.powf(exp) }
}

/// Standard interface for frequency analysis tools
pub trait FrequencyTools<T: RealField, const N: usize, const M: usize> {
    /// Calculates the complex response from a set of input frequencies
    fn frequency_response<const L: usize>(&self, response: &mut FrequencyResponse<T, L, N, M>);
}

/// A unified Frequency Response object
///
/// This struct provides constructors and utilities for generating and handling frequency response
/// data for a system with generic input and output dimensions. The number of frequency points is
/// also configured through a generic parameter to allow easily controlling the resolution.
///
/// > A channel is a path from input to output, a system's channels are commonly represented as a
/// > matrix (with rows = inputs and columns = outputs).
///
/// By default, the constructors will set the response to `None` and only fill the frequencies that
/// will be sampled.
///
/// # Generic Arguments
/// * `T` - The field type for frequencies and responses
/// * `L` - The number of frequency points to sample per channel
/// * `N` - Dimension of the system's input
/// * `M` - Dimension of the system's output
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrequencyResponse<T, const L: usize, const N: usize, const M: usize> {
    /// A 2D array where each row corresponds to the frequency points for a single input channel.
    pub frequencies: [[T; L]; N],
    /// A 2D array where each row corresponds to the complex frequency responses for a single
    /// output channel.
    pub responses: Option<[[Complex<T>; L]; M]>,
}

impl<T, const L: usize, const N: usize, const M: usize> FrequencyResponse<T, L, N, M> {
    /// Creates a new `FrequencyResponse` with specified frequency data and zeroed responses.
    ///
    /// # Arguments
    /// * `frequencies` - A 2D array containing frequency points for each input channel
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with the provided frequencies and zero-initialized
    ///   responses
    pub const fn new(frequencies: [[T; L]; N]) -> Self {
        Self {
            frequencies,
            responses: None,
        }
    }
}

impl<T, const L: usize, const N: usize, const M: usize> FrequencyResponse<T, L, N, M>
where
    T: Copy + Zero + RealField,
{
    /// Extract the magnitude and phase from a frequency responses output channel
    ///
    /// If there is no frequency response, the magnitude and phase arrays will be zero-filled
    pub fn mag_phase(&self, output_channel: usize) -> ([T; L], [T; L]) {
        // TODO: remove zero initialization + move this to general function so Margins can reuse it
        let mut mag = [T::zero(); L];
        let mut phase = [T::zero(); L];
        if let Some(responses) = self.responses {
            responses[output_channel]
                .iter()
                .zip(mag.iter_mut().zip(phase.iter_mut()))
                .for_each(|(response, (mag, phase))| (*mag, *phase) = response.to_polar());
        }
        (mag, phase)
    }
}

impl<T, const L: usize, const N: usize, const M: usize> FrequencyResponse<T, L, N, M>
where
    T: Float + AddAssign,
{
    /// Creates a new `FrequencyResponse` with logarithmically spaced frequencies.
    ///
    /// # Arguments
    /// * `freq_start` - Start frequencies for each input channel
    /// * `freq_stop` - Stop frequencies for each input channel
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with logarithmically spaced frequencies and
    ///   no responses
    pub fn logspace(freq_start: [T; N], freq_stop: [T; N]) -> Self {
        // Safety: There will be N arrays of L log spaced values
        let frequencies = unsafe {
            array_from_iterator(
                (0..N).map(|i| logspace::<T, L>(freq_start[i], freq_stop[i])),
            )
        };
        Self {
            frequencies,
            responses: None,
        }
    }

    /// Creates a new `FrequencyResponse` where only one input channel has frequency data.
    ///
    /// # Arguments
    /// * `freq_start` - The start frequency for the isolated input channel
    /// * `freq_stop` - The stop frequency for the isolated input channel
    /// * `channel` - The index of the input channel to fill
    ///
    /// # Returns
    /// * `FrequencyResponse` instance with frequency data for the specified channel and no responses
    pub fn isolated(freq_start: T, freq_stop: T, channel: usize) -> Self {
        // TODO: remove zero initialization, or should this whole function get removed
        let mut frequencies = [[T::zero(); L]; N];
        frequencies[channel] = logspace(freq_start, freq_stop);
        Self {
            frequencies,
            responses: None,
        }
    }

    /// Creates a new `FrequencyResponse` where all input channels share the same frequency data
    ///
    /// This function will generate N copies of a log space between the given start and stop
    /// frequencies.
    ///
    /// # Arguments
    /// * `freq_start` - The start frequency for all input channels
    /// * `freq_stop` - The stop frequency for all input channels
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with shared frequency data across all input channels and zero-initialized responses
    pub fn simultaneous(freq_start: T, freq_stop: T) -> Self {
        Self {
            frequencies: [logspace(freq_start, freq_stop); N],
            responses: None,
        }
    }
}

/// A structure representing frequency stability margins.
///
/// This struct encapsulates key frequency domain metrics used to evaluate the stability
/// of a control system, such as phase crossover, gain crossover, phase margin, and gain margin.
///
/// # Generic Arguments
/// - `T`: The numeric type for the stability metrics (e.g., `f32` or `f64`).
/// 
/// TODO: This shouldn't need to store the margin, those should be easily extracted from the resp
#[derive(Copy, Clone, Debug)]
pub struct PhaseGainCrossover<T> {
    /// The frequency at which the phase crosses -180 degrees
    pub phase_crossover: Option<T>,
    /// The frequency at which the gain crosses 0 dB (unity gain)
    pub gain_crossover: Option<T>,
    /// The margin between the actual phase and -180 degrees at the gain crossover frequency
    pub phase_margin: Option<T>,
    /// The margin between the actual gain and 0 dB at the phase crossover frequency
    pub gain_margin: Option<T>,
}

impl<T: Copy + Zero + One + RealField + Phase + Magnitude> PhaseGainCrossover<T> {
    /// Computes both the magnitude and phase crossover frequencies and margins of
    /// a frequency response
    ///
    /// # Arguments
    /// * `frequencies` - The array of frequency values (in rad/s)
    /// * `response` - The array of `Complex<T>` corresponding to the given frequencies
    ///
    /// # Returns
    /// * `PhaseGainCrossover` - the margins and crossovers of the response
    pub fn new<const L: usize>(frequencies: &[T; L], response: &[Complex<T>; L]) -> Self {
        // TODO: Remove duplicate code + see `FrequencyResponse::mag_phase()`
        let mut phases = [T::zero(); L];
        let mut magnitudes = [T::zero(); L];

        (0..L).for_each(|i| (magnitudes[i], phases[i]) = response[i].to_polar());

        let gain_crossover = first_crossover(frequencies, &magnitudes, T::one());
        let phase_crossover = first_crossover(frequencies, &phases, -T::PI);

        let gain_margin = phase_crossover.map(|wc| {
            // should be using frequencies.iter().position(|hz| hz == threshold, or change first_crossover to return the index)
            // unwrap is safe because the crossover frequency exists, meaning there is a corresponding
            // index in the magnitude array
            // TODO: Fix this so there doesn't need to be an unwrap (see `first_crossover()` todos)
            let mag_at_crossover =
                first_crossover(&magnitudes, frequencies, wc).unwrap_or_else(|| T::zero());
            mag_at_crossover.to_db()
        });

        let phase_margin = gain_crossover.map(|wc| {
            // unwrap is safe because the crossover frequency exists, meaning there is a corresponding
            // index in the phase array
            let phase_at_crossover =
                first_crossover(&phases, frequencies, wc).unwrap_or_else(|| T::zero());
            (phase_at_crossover + T::PI).to_degrees()
        });

        Self {
            phase_crossover,
            gain_crossover,
            phase_margin,
            gain_margin,
        }
    }
}
impl<T> Default for PhaseGainCrossover<T> {
    /// Creates a `PhaseGainCrossover` full of None
    /// # Returns
    /// * `PhaseGainCrossover` - with all fields set to `None`
    fn default() -> Self {
        Self {
            phase_crossover: None,
            gain_crossover: None,
            phase_margin: None,
            gain_margin: None,
        }
    }
}

/// A structure representing stability margins for every input-output channel of a system
///
/// This struct provides a matrix of `PhaseGainCrossover` values, where each element corresponds
/// to the stability margins for a specific input-output channel pair.
///
/// # Generic Arguments
/// - `T`: Field type for the stability metrics (e.g., `f32` or `f64`).
/// - `N`: Dimension of the system's input.
/// - `M`: Dimension of the system's output.
pub struct FrequencyMargin<T, const N: usize, const M: usize>(pub [[PhaseGainCrossover<T>; N]; M]);

impl<T: Copy + Zero + One + RealField + Magnitude + Phase, const N: usize, const M: usize> FrequencyMargin<T, N, M> {
    /// Creates a new `FrequencyMargin` instance from a given `FrequencyResponse`
    ///
    /// This function computes the stability margins for all input-output channel pairs in the
    /// provided `FrequencyResponse` object.
    ///
    /// # Arguments
    /// * `response` - A reference to a `FrequencyResponse` containing the frequency and response
    ///   data
    ///
    /// # Generic Arguments
    /// * `L` - The number of frequency points per channel
    ///
    /// # Returns
    /// * `FrequencyMargin` - A new instance containing the calculated margins for all input-output channels
    pub fn new<const L: usize>(response: &FrequencyResponse<T, L, N, M>) -> Self {
        let mut margins = [[PhaseGainCrossover::default(); N]; M];
        if let Some(responses) = response.responses {
            for (margin_row, res) in margins.iter_mut().zip(responses.iter()) {
                for (margin, frequency) in margin_row.iter_mut().zip(response.frequencies.iter()) {
                    *margin = PhaseGainCrossover::new::<L>(frequency, res);
                }
            }
        }
        Self(margins)
    }
}

/// Computes the first crossover of a signal
///
/// Given two 1D arrays and a threshold, this function will find the value of the first when the
/// second crosses the given threshold:
///
/// <pre>b[i] < thresh < b[i+1] || b[i] > thresh > b[i+1]</pre>
///
/// Used to find the crossover frequencies of a transfer function's gain and phase.
/// The gain crossover frequency is the frequency at which the gain of the transfer
/// function is 1 (0 dB). The phase crossover frequency is where the phase crosses -180°.
///
/// If no crossover is found within the given frequency range, the function returns `None` for
/// that parameter.
///
/// # Arguments
/// * `a` - The first array of values (the return value is sampled or interpolated from these)
/// * `b` - The second array of values (these are the values that may cross the threshold)
/// * `threshold` - The target threshold for the crossover (e.g., 1.0 for magnitude, -π for phase)
///
/// # Generic Arguments
/// * `T` - Scalar type for frequencies and values (e.g., `f32`, `f64`)
/// * `N` - Number of frequency samples provided
///
/// # Returns
/// * `Option<T>` - The crossover frequency in rad/s, or `None` if no crossover frequency is found
/// 
/// TODO: 
///   * Split this up into `first_crossover(arr, threshold) -> index` and
///     `interpolate_arrays(arr_a, arr_b) -> T`
fn first_crossover<T, const N: usize>(a: &[T; N], b: &[T; N], threshold: T) -> Option<T>
where
    T: Clone + PartialOrd + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    for i in 0..N - 1 {
        if b[i] == threshold {
            return Some(a[i].clone());
        }
        if (b[i] < threshold && b[i + 1] > threshold) || (b[i] > threshold && b[i + 1] < threshold)
        {
            let slope = (b[i + 1].clone() - b[i].clone()) / (a[i + 1].clone() - a[i].clone());
            let intercept = b[i].clone() - slope.clone() * a[i].clone();
            return Some((threshold - intercept) / slope);
        }
    }
    // need some cleanup here
    if b[N - 1] == threshold {
        return Some(a[N - 1].clone());
    }
    None
}

/// Generates `N` logarithmically spaced points of type `T` between 10^a and 10^b.
///
/// # Type Parameters
/// - `T`: A floating-point type implementing `num_traits::Float` (e.g., `f32`, `f64`)
/// - `N`: Number of points (compile-time constant)
///
/// # Example
/// ```
/// use control_rs::frequency_tools::logspace;
/// let points = logspace::<f64, 5>(1.0, 3.0);
/// assert_eq!(points.len(), 5);
/// ```
/// # Panics
/// * if 10.0 cannot cast to T
/// * if N - 1 cannot cast to T
/// 
/// TODO: remove unwraps + make a 10E struct that impl powf for floats and ints
pub fn logspace<T: Float + AddAssign, const N: usize>(a: T, b: T) -> [T; N] {
    let mut result = [T::zero(); N];
    #[allow(clippy::unwrap_used)]
    let ten = T::from(10.0).unwrap();

    // Edge case: one point
    if N == 1 {
        result[0] = ten.powf(a);
        return result;
    }

    #[allow(clippy::unwrap_used)]
    let step = (b - a) / T::from(N - 1).unwrap();

    let mut exponent = a;
    for r in &mut result {
        *r = ten.powf(exponent);
        exponent += step;
    }

    result
}

#[cfg(feature = "std")]
/// Renders a single magnitude and phase plot on subplots
fn render_bode_subplot<T>(
    plot: &mut plotly::Plot,
    frequencies: &[T],
    mag: &[T],
    phase: &[T],
    row: usize,
    col: usize,
    margins: &PhaseGainCrossover<T>,
) where
    T: 'static + Copy + Neg<Output = T> + Zero + One + Magnitude + Phase + serde::ser::Serialize,
{
    use plotly::common;
    extern crate std;
    use std::{format, vec, vec::Vec};

    let mag_db: Vec<T> = mag
        .iter()
        .map(|&m| m.to_db())
        .collect();
    let phase_deg: Vec<T> = phase.iter().map(|&p| p.to_degrees()).collect();

    // Add magnitude plot
    plot.add_trace(
        plotly::Scatter::new(frequencies.to_vec(), mag_db)
            .mode(common::Mode::Lines)
            .name(format!("Magnitude[{row}, {col}]"))
            // .x_axis(x_axis_mag.clone())
            // .y_axis(y_axis_mag.clone()).color(Rgb::new(0, 0, 255))
            .marker(common::Marker::new()),
    );

    // Add phase plot
    plot.add_trace(
        plotly::Scatter::new(frequencies.to_vec(), phase_deg)
            .mode(common::Mode::Lines)
            .name(format!("Phase[{row}, {col}]"))
            .x_axis("x2")
            .y_axis("y2")
            .marker(common::Marker::new()),
    );

    // Gain margin line
    if let (Some(wc), Some(gm)) = (margins.phase_crossover, margins.gain_margin) {
        plot.add_trace(
            plotly::Scatter::new(vec![wc, wc], vec![-gm, T::zero()])
                .mode(common::Mode::Lines)
                .name(format!("Gain FrequencyMargin[{row}, {col}]"))
                // .x_axis(x_axis_mag)
                // .y_axis(y_axis_mag)
                .line(
                    common::Line::new()
                        .dash(common::DashType::Dot)
                        .color(plotly::color::Rgb::new(0, 0, 0)),
                ),
        );
    }

    // Phase margin line
    if let (Some(wc), Some(pm)) = (margins.gain_crossover, margins.phase_margin) {
        plot.add_trace(
            plotly::Scatter::new(vec![wc, wc], vec![T::ONE_EIGHTY.neg(), pm])
                .mode(common::Mode::Lines)
                .name(format!("Phase FrequencyMargin[{row}, {col}]"))
                .x_axis("x2")
                .y_axis("y2")
                .line(
                    common::Line::new()
                        .dash(common::DashType::Dot)
                        .color(plotly::color::Rgb::new(0, 0, 0)),
                ),
        );
    }
}

#[cfg(feature = "std")]
/// Renders a Bode plot for an object implementing `FrequencyTools`
pub fn bode<T, F, const L: usize, const N: usize, const M: usize>(
    title: &str,
    system: &F,
    mut response: FrequencyResponse<T, L, N, M>,
) -> plotly::Plot
where
    T: Copy + Neg<Output = T> + Zero + One + RealField + Magnitude + Phase + serde::ser::Serialize,
    F: FrequencyTools<T, N, M>,
{
    use plotly::Layout;

    if response.responses.is_none() {
        system.frequency_response(&mut response);
    }
    let margins = FrequencyMargin::new(&response);

    let mut plot = plotly::Plot::new();
    plot.set_layout(
        Layout::new()
            .title(plotly::common::Title::with_text(title))
            .x_axis(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::with_text("Frequency (rad/s)"))
                    .type_(plotly::layout::AxisType::Log),
            )
            .y_axis(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::with_text("Magnitude (dB)")),
            )
            .x_axis2(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::with_text("Frequency (rad/s)"))
                    .type_(plotly::layout::AxisType::Log),
            )
            .y_axis2(
                plotly::layout::Axis::new().title(plotly::common::Title::with_text("Phase (deg)")),
            )
            .grid(
                plotly::layout::LayoutGrid::new()
                    .rows(2)
                    .columns(1)
                    .pattern(plotly::layout::GridPattern::Independent)
                    .row_order(plotly::layout::RowOrder::TopToBottom),
            ),
    );

    // extract each output mag/phase (make a helper in response soon)
    for out_channel in 0..M {
        let (magnitudes, phases) = response.mag_phase(out_channel);

        // render the output as the response to each input
        for (in_channel, frequency) in response.frequencies.iter().enumerate() {
            render_bode_subplot(
                &mut plot,
                frequency,
                &magnitudes,
                &phases,
                in_channel,
                out_channel,
                &margins.0[out_channel][in_channel],
            );
        }
    }

    plot
}

#[cfg(test)]
mod test_log_space {
    use super::*;
    use crate::assert_f32_eq;

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn basic() {
        let ls = logspace::<f32, 10>(1.0, 5.0);
        assert_f32_eq!(ls[0], 10.0);
        assert_f32_eq!(ls[1], 30.0, 4.0);
        assert_f32_eq!(ls[2], 80.0, 4.0);
        assert_f32_eq!(ls[3], 220.0, 5.0);
        assert_f32_eq!(ls[4], 600.0, 1.0);
        assert_f32_eq!(ls[5], 1670.0, 2.0);
        assert_f32_eq!(ls[6], 4640.0, 2.0);
        assert_f32_eq!(ls[7], 12920.0, 5.0);
        assert_f32_eq!(ls[8], 35940.0, 2.0);
        assert_f32_eq!(ls[9], 100_000.0, 10.0);
    }
}

#[cfg(test)]
mod test_first_crossover {
    use super::*;
    use crate::{assert_f32_eq, assert_f64_eq};

    #[test]
    fn crossover_detection() {
        // Test inputs
        let frequencies: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
        let threshold = 1.0;

        // Trigger edge case where b[i] == threshold
        let result = first_crossover(&frequencies, &magnitudes, threshold);

        assert!(result.is_some());
        #[allow(clippy::unwrap_used)]
        let result = result.unwrap();
        assert_f64_eq!(result, 1.0); // `frequencies[2]`
    }

    #[test]
    fn crossover_interpolation() {
        // Test inputs
        let frequencies: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.9, 0.95, 1.05, 1.2, 1.3]; // inconsistent step
        let threshold = 1.0;

        // Linear interpolation for crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);

        assert!(result.is_some());
        #[allow(clippy::unwrap_used)]
        let result = result.unwrap();
        assert_f64_eq!(result, 1.25, 1e-10); // Interpolated value
    }

    #[test]
    fn no_crossover() {
        // Test inputs with no crossover
        let frequencies: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99];
        let threshold = 1.0;

        // No crossover detected
        let result = first_crossover(&frequencies, &magnitudes, threshold);

        assert!(result.is_none());
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn decreasing_crossover() {
        // Test inputs with decreasing crossover
        let frequencies: [f32; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f32; 6] = [1.2, 1.1, 1.05, 0.95, 0.8, 0.7];
        let threshold = 1.0;

        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);

        assert_f32_eq!(result.unwrap(), 1.25, 1.2e-7); // Interpolated value
    }
    #[allow(clippy::unwrap_used)]
    #[test]
    fn multiple_threshold_instances() {
        let frequencies: [usize; 10] = core::array::from_fn(|i| i);
        let magnitudes: [usize; 10] = core::array::from_fn(|i| (i >= 5).into());
        let threshold = 1;
        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);
        assert_eq!(result.unwrap(), 5); // Interpolated value
    }
    #[allow(clippy::unwrap_used, clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    #[test]
    fn integer_interpolation() {
        let frequencies: [i32; 10] = core::array::from_fn(|i| i as i32);
        let magnitudes: [i32; 10] = core::array::from_fn(|i| if i < 5 { 5 } else { 7 });
        let threshold = 6;
        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);
        assert_eq!(result.unwrap(), 4); // Interpolated value
    }

    #[allow(clippy::unwrap_used, clippy::cast_possible_wrap, clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    #[test]
    fn multiple_crossovers() {
        let frequencies: [f32; 10] = core::array::from_fn(|i| i as f32);
        let magnitudes: [f32; 10] = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { 3.0 });
        let threshold = 2.0;
        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);
        assert_f32_eq!(result.unwrap(), 0.5); // Interpolated value
    }
}
