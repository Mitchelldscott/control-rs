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
//! - [ ] move plotly to plotly helper file (or wait for a nice gui)

#[cfg(feature = "std")]
use core::ops::Neg;
use core::ops::{AddAssign, Div, Mul, Sub};

use nalgebra::{Complex, ComplexField, RealField};
use num_traits::{Float, One, Zero};

use crate::static_storage::arrays_from_zipped_iterator;

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
    fn to_radians(self) -> Self {
        self * (Self::PI / Self::ONE_EIGHTY)
    }
    /// Converts the angle from radians to degrees.
    #[must_use]
    #[inline]
    fn to_degrees(self) -> Self {
        self * (Self::ONE_EIGHTY / Self::PI)
    }
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
    fn to_db(self) -> Self {
        Self::TWENTY * self.log10()
    }

    /// Converts a decibel (dB) value to an absolute magnitude.
    ///
    /// The formula used is `DB_REFERENCE * 10^(self / DB_MULTIPLIER)`.
    #[must_use]
    #[inline]
    fn to_mag(self) -> Self {
        Self::TEN.powf(self / Self::TWENTY)
    }
}

impl Magnitude for f32 {
    const TEN: Self = 10.0;
    const TWENTY: Self = 20.0;
    #[inline]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl Magnitude for f64 {
    const TEN: Self = 10.0;
    const TWENTY: Self = 20.0;
    #[inline]
    fn log10(self) -> Self {
        self.log10()
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

/// Standard interface for frequency analysis tools
pub trait FrequencyTools<T, const N: usize, const M: usize> {
    /// Calculates the complex response from a set of input frequencies
    fn frequency_response<const K: usize>(&self, response: &mut FrequencyResponse<T, N, M, K>);
}

/// A Unified Frequency Response.
///
/// This struct provides constructors and utilities for generating and handling frequency response
/// data for a system with generic input and output channels.
///
/// By default, the constructors will set the response to `None`.
///
/// # Generic Arguments
/// * `T` - The field type for frequencies and responses.
/// * `N` - The number of input channels.
/// * `M` - The number of output channels.
/// * `K` - The number of frequency points to be sampled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrequencyResponse<T, const N: usize, const M: usize, const K: usize> {
    /// An array of the frequency points to sample every input and output channel.
    pub frequencies: [T; K],
    /// A 3D array representing the complex frequency responses. The dimensions are:
    /// `[input_index][output_index][frequency_point_index]`. This means for each of the `K`
    /// frequency points, there are `MxN` complex responses, where `responses[i][j][k]`
    /// is the response from input `i` to output `j` at the `k`-th frequency point.
    pub responses: Option<[[[Complex<T>; K]; M]; N]>,
}

impl<T, const N: usize, const M: usize, const K: usize> FrequencyResponse<T, N, M, K> {
    /// Creates a new `FrequencyResponse` with specified frequency data and no responses.
    ///
    /// # Arguments
    /// * `frequencies` - An array containing frequency points.
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with the provided frequencies and zero-initialized
    ///   responses.
    pub const fn new(frequencies: [T; K]) -> Self {
        Self {
            frequencies,
            responses: None,
        }
    }
}

impl<T, const N: usize, const M: usize, const K: usize> FrequencyResponse<T, N, M, K>
where
    T: RealField,
{
    /// Extract the magnitude and phase from a frequency response channel.
    ///
    /// # Arguments
    /// * `input_channel` - index of the input channel to extract a response from.
    /// * `output_channel` - index of the output channel to extract a response from.
    ///
    ///# Returns
    /// * `Option`
    ///     * `(mag, phase)` - magnitude and phase of the specified input/output response channel.
    ///     * `None` - There is no response data for the specified channel.
    pub fn mag_phase(
        &self,
        input_channel: usize,
        output_channel: usize,
    ) -> Option<([T; K], [T; K])> {
        if input_channel < N
            && output_channel < M
            && let Some(responses) = &self.responses
        {
            Some(unzip_complex_array(
                &responses[input_channel][output_channel],
            ))
        } else {
            None
        }
    }
}

impl<T, const N: usize, const M: usize, const K: usize> FrequencyResponse<T, N, M, K>
where
    T: Float + AddAssign,
{
    /// Creates a new `FrequencyResponse` with logarithmically spaced frequencies.
    ///
    /// This is a wrapper for [`logspace()`].
    ///
    /// # Arguments
    /// * `freq_start` - Start frequency.
    /// * `freq_stop` - Stop frequencies.
    ///
    /// # Returns
    /// * `FrequencyResponse` - Instance with logarithmically spaced frequencies and
    ///   no responses.
    pub fn logspace(freq_start: T, freq_stop: T) -> Self {
        Self {
            frequencies: logspace(freq_start, freq_stop),
            responses: None,
        }
    }
}

/// A structure representing frequency stability margins and their corresponding crossover
/// frequencies.
///
/// This struct encapsulates key frequency domain metrics used to evaluate the stability
/// of a control system, such as phase crossover, gain crossover, phase margin, and gain margin.
///
/// # Generic Arguments
/// - `T`: Field type for the stability metrics.
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
    pub fn new<const K: usize>(frequencies: &[T; K], response: &[Complex<T>; K]) -> Self {
        let (magnitudes, phases) = unzip_complex_array(response);
        let (gain_margin, phase_crossover) = find_margin(&phases, &magnitudes, frequencies, -T::PI);
        let (phase_margin, gain_crossover) =
            find_margin(&magnitudes, &phases, frequencies, T::one());

        Self {
            phase_crossover,
            gain_crossover,
            phase_margin: phase_margin.map(|rads| (rads + T::PI).to_degrees()),
            gain_margin: gain_margin.map(|mag| mag.to_db().neg()),
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

impl<T: Copy + Zero + One + RealField + Magnitude + Phase, const N: usize, const M: usize>
    FrequencyMargin<T, N, M>
{
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
    pub fn new<const K: usize>(frequency_response: &FrequencyResponse<T, N, M, K>) -> Self {
        let mut margins = [[PhaseGainCrossover::default(); N]; M];
        if let Some(responses) = frequency_response.responses {
            for (margin_row, response_row) in margins.iter_mut().zip(responses.iter()) {
                for (margin, response) in margin_row.iter_mut().zip(response_row.iter()) {
                    *margin = PhaseGainCrossover::new(&frequency_response.frequencies, response);
                }
            }
        }
        Self(margins)
    }
}

/// Helper function to unzip an array of complex numbers into an array of magnitudes and phases
///
/// # Arguments
/// * `complex` - Array of complex values.
///
/// # Returns
/// * `(magnitude, phase)` - Tuple of magnitudes and phases.
///
/// # Safety
/// This function makes an `unsafe` call to initialize two arrays from a collection of tuples.
/// The call is guaranteed to be safe because there are K elements that will get mapped to K tuples.
pub fn unzip_complex_array<T: RealField, const K: usize>(
    complex_array: &[Complex<T>; K],
) -> ([T; K], [T; K]) {
    // Safety: `complex_array` has exactly K elements, `to_polar()` converts each of those elements
    // to a tuple of (T, T) so all elements of both arrays are guaranteed to get fully initialized.
    unsafe {
        arrays_from_zipped_iterator(
            complex_array
                .iter()
                .map(|complex| complex.clone().to_polar()),
        )
    }
}

/// Finds the value in two arrays where a third array crosses a threshold.
///
/// The function will attempt to interpolate the values in the second and third array using a linear
/// approximation of the crossover in the first array.
///
/// # Generic Arguments
/// * `T` - Field type of the threshold and three input arrays.
/// * `K` - Capacity of each array.
///
/// # Arguments
/// * `crossover` - The array that may cross the threshold.
/// * `a` - First array to read a value from.
/// * `b` - Second array to read a value from.
///
/// # Returns
/// * `(Option, Option)`
///     * `(value_a, value_b)` - The approximate values from the second and third arrays.
///     * `(None, None)` - The crossover array does not cross the threshold.
pub fn find_margin<T, const K: usize>(
    crossover: &[T; K],
    a: &[T; K],
    b: &[T; K],
    threshold: T,
) -> (Option<T>, Option<T>)
where
    T: Clone + PartialOrd + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    if let Some((index, is_exact)) = first_crossover_index(crossover, &threshold) {
        if is_exact {
            (Some(a[index].clone()), Some(b[index].clone()))
        } else {
            (
                Some(array_interpolation(crossover, a, index, threshold.clone())),
                Some(array_interpolation(crossover, b, index, threshold)),
            )
        }
    } else {
        (None, None)
    }
}
/// Computes the first crossover of a signal.
///
/// This function will find the index when the array crosses the given threshold:
///
/// <pre>b[i] < thresh < b[i+1] || b[i] > thresh > b[i+1]</pre>
///
/// If no crossover is found within the given frequency range, the function returns `None`.
///
/// # Generic Arguments
/// * `T` - Field type of the array and threshold.
/// * `K` - Capacity of the array.
///
/// # Arguments
/// * `array` - The array of values to compare with the threshold.
/// * `threshold` - The target threshold for the crossover.
///
/// # Returns
/// * `Option`
///     * `(index, exact_crossover)` - Index of the first crossover and a bool if the crossover was
///       exactly at that index.
///     * `None` - The array never crosses the threshold.
fn first_crossover_index<T, const K: usize>(array: &[T; K], threshold: &T) -> Option<(usize, bool)>
where
    T: PartialOrd,
{
    for i in 0..K - 1 {
        if array[i] == *threshold {
            return Some((i, true));
        }
        if (array[i] < *threshold && array[i + 1] > *threshold)
            || (array[i] > *threshold && array[i + 1] < *threshold)
        {
            return Some((i, false));
        }
    }
    if array[K - 1] == *threshold {
        return Some((K - 1, true));
    }
    None
}

/// Computes the value in array 'b' at the point where array 'a' crosses a threshold.
///
/// This function is designed to be used in conjunction with `find_crossover_index`.
///
/// # Generic Arguments
/// * `T` - Field type of the array and threshold.
/// * `K` - Capacity of the array.
///
/// # Arguments
/// * `a` - The array of values that crossed the threshold.
/// * `b` - The array to interpolate.
/// * `index` - The target threshold for the interpolation.
/// * `threshold` - The target value of the crossover.
///
/// # Returns
/// * `interpolated_value` - The interpolated value from `b`.
///
/// # Panics
/// * If index == K-1, out-of-bounds access will occur.
fn array_interpolation<T, const K: usize>(a: &[T; K], b: &[T; K], index: usize, threshold: T) -> T
where
    T: Clone + PartialOrd + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    // Perform linear interpolation
    // We have two points: (a[index], b[index]) and (a[index+1], b[index+1])
    // We want to find 'x' (from array 'b') when 'y' (from array 'a') is 'threshold'.
    // The equation of a line is y = m*x + c
    // where m = (y2 - y1) / (x2 - x1)
    // and c = y1 - m*x1
    // We want x = (y - c) / m
    if a[index].clone() == a[index + 1].clone() || b[index].clone() == b[index + 1].clone() {
        return b[index].clone();
    }

    // Calculate the slope of 'a' with respect to 'b'
    let slope =
        (a[index + 1].clone() - a[index].clone()) / (b[index + 1].clone() - b[index].clone());

    // Calculate the y-intercept (for the line y = m*x + c)
    let intercept = a[index].clone() - slope.clone() * b[index].clone();

    // Solve for x (the value from 'b') when y (the value from 'a') is 'threshold'
    (threshold - intercept) / slope
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
/// TODO: remove unwraps + make a 10E trait that impl powf for floats and ints
pub fn logspace<T: Float + AddAssign, const N: usize>(a: T, b: T) -> [T; N] {
    let mut result = [T::zero(); N];

    #[allow(clippy::unwrap_used)]
    let ten = T::from(10).unwrap();

    // Edge case: one point
    if N == 1 {
        result[0] = ten.powf(a);
        return result;
    }

    // TODO: Remove this unwrap
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

    let mag_db: Vec<T> = mag.iter().map(|&m| m.to_db()).collect();
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
            plotly::Scatter::new(vec![wc, wc], vec![T::ONE_EIGHTY.neg(), T::ONE_EIGHTY.neg()+pm])
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
pub fn bode<T, F, const N: usize, const M: usize, const K: usize>(
    title: &str,
    system: &F,
    mut response: FrequencyResponse<T, N, M, K>,
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
    for in_channel in 0..N {
        for out_channel in 0..M {
            if let Some((magnitudes, phases)) = response.mag_phase(in_channel, out_channel) {
                // render the output as the response to each input
                render_bode_subplot(
                    &mut plot,
                    &response.frequencies,
                    &magnitudes,
                    &phases,
                    in_channel,
                    out_channel,
                    &margins.0[in_channel][out_channel],
                );
            }
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
mod test_find_margin {
    use super::*;
    use crate::{assert_f32_eq, assert_f64_eq};

    #[test]
    fn crossover_detection() {
        // Test inputs
        let frequencies = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
        let phases = [-f64::PI; 6];
        let threshold = 1.0;

        // Trigger edge case where b[i] == threshold
        let margin = find_margin(&magnitudes, &phases, &frequencies, threshold);

        assert!(margin.0.is_some());
        #[allow(clippy::unwrap_used)]
        let phase_margin = margin.0.unwrap();
        assert_f64_eq!(phase_margin, -f64::PI);

        assert!(margin.1.is_some());
        #[allow(clippy::unwrap_used)]
        let gain_crossover = margin.1.unwrap();
        assert_f64_eq!(gain_crossover, 1.0);
    }

    #[test]
    fn crossover_interpolation() {
        // Test inputs
        let frequencies = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes = [0.8, 0.9, 0.95, 1.05, 1.2, 1.3]; // inconsistent step
        let phases = [-f64::PI; 6];
        let threshold = 1.0;

        let margin = find_margin(&magnitudes, &phases, &frequencies, threshold);

        assert!(margin.0.is_some());
        #[allow(clippy::unwrap_used)]
        let phase_margin = margin.0.unwrap();
        assert_f64_eq!(phase_margin, -f64::PI);

        assert!(margin.1.is_some());
        #[allow(clippy::unwrap_used)]
        let gain_crossover = margin.1.unwrap();
        assert_f64_eq!(gain_crossover, 1.25, 1e-10);
    }

    #[test]
    fn no_crossover() {
        // Test inputs with no crossover
        let frequencies: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99];
        let phases = [-f64::PI; 6];
        let threshold = 1.0;

        // No crossover detected
        let margin = find_margin(&magnitudes, &phases, &frequencies, threshold);

        assert!(margin.0.is_none());
        assert!(margin.1.is_none());
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn decreasing_crossover() {
        // Test inputs with decreasing crossover
        let frequencies: [f32; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f32; 6] = [1.2, 1.1, 1.05, 0.95, 0.8, 0.7];
        let threshold = 1.0;

        // Linear interpolation for decreasing crossover
        let index = first_crossover_index(&magnitudes, &threshold).unwrap();
        assert!(!index.1);
        let result = array_interpolation(&magnitudes, &frequencies, index.0, threshold);
        assert_f32_eq!(result, 1.25, 1.2e-7); // Interpolated value
    }
    #[allow(clippy::unwrap_used)]
    #[test]
    fn multiple_threshold_instances() {
        let frequencies: [usize; 10] = core::array::from_fn(|i| i);
        let magnitudes: [usize; 10] = core::array::from_fn(|i| (i >= 5).into());
        let threshold = 1;
        let index = first_crossover_index(&magnitudes, &threshold).unwrap();
        assert!(index.1);
        assert_eq!(frequencies[index.0], 5);
    }
    #[allow(
        clippy::unwrap_used,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation
    )]
    #[test]
    fn integer_interpolation() {
        let frequencies: [i32; 10] = core::array::from_fn(|i| i as i32);
        let magnitudes: [i32; 10] = core::array::from_fn(|i| if i < 5 { 5 } else { 7 });
        let threshold = 6;
        let index = first_crossover_index(&magnitudes, &threshold).unwrap();
        assert!(!index.1);
        assert_eq!(frequencies[index.0], 4);
    }

    #[allow(
        clippy::unwrap_used,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss
    )]
    #[test]
    fn multiple_crossovers() {
        let frequencies: [f32; 10] = core::array::from_fn(|i| i as f32);
        let magnitudes: [f32; 10] = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { 3.0 });
        let threshold = 2.0;
        let index = first_crossover_index(&magnitudes, &threshold).unwrap();
        assert!(!index.1);
        let result = array_interpolation(&magnitudes, &frequencies, index.0, threshold);
        assert_f32_eq!(result, 0.5);
    }
    #[test]
    #[should_panic = "index out of bounds: the len is 6 but the index is 6"]
    fn array_interpolation_oob() {
        let frequencies: [usize; 6] = core::array::from_fn(|i| i);
        let magnitudes: [usize; 6] = core::array::from_fn(|i| (i >= 5).into());
        array_interpolation(&magnitudes, &frequencies, 5, 1);
    }
}
