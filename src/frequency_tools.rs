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
//! - [ ] docs, docs, docs
//! - [ ] move [Margin] -> [`FrequencyResponse`], no point in having two structs using the same generics
//!     - add margins function to FR
//!     - add `to_polar()` to FR (unrelated but useful, maybe make a `PolarFrequencyResponse`)
//! - [ ] textbook example of trait productivity
//! - [ ] move plotly to plotly helper file (or wait for a nice gui)
//! - [ ] move FR constructors to FR factory? (configure set of FRs and target outputs without initializing anything)

use core::ops::{AddAssign, Sub, Mul, Div};
use nalgebra::{Complex, ComplexField, RealField};
use num_traits::Float;

/// Standard interface for frequency analysis tools
pub trait FrequencyTools<T: Float + RealField, const N: usize, const M: usize> {
    /// Calculates the complex response from a set of input frequencies
    fn frequency_response<const L: usize>(&self, response: &mut FrequencyResponse<T, L, N, M>);
}

/// A unified Frequency Response object
///
/// This struct provides constructors and utilities for generating and handling
/// frequency response data for a system with multiple input and output channels.
///
/// # Generic Arguments
/// * `T` - The numeric type for frequencies and responses (e.g., `f32` or `f64`)
/// * `L` - The number of frequency points per channel
/// * `N` - The number of input channels
/// * `M` - The number of output channels
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyResponse<T, const L: usize, const N: usize, const M: usize> {
    /// Frequencies for each input channel
    ///
    /// A 2D array where each row corresponds to the frequency points for a single input channel.
    pub frequencies: [[T; L]; N],
    /// Response of each output channel for each input frequency
    ///
    /// A 2D array where each row corresponds to the complex frequency responses for a single output channel.
    pub responses: [[Complex<T>; L]; M],
}

impl<T: Float + AddAssign, const L: usize, const N: usize, const M: usize>
    FrequencyResponse<T, L, N, M>
{
    /// Creates a new `FrequencyResponse` with specified frequency data and zeroed responses.
    ///
    /// # Arguments
    /// * `frequencies` - A 2D array containing frequency points for each input channel
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with the provided frequencies and zero-initialized
    ///   responses
    pub fn new(frequencies: [[T; L]; N]) -> Self {
        let responses = [[Complex::new(T::zero(), T::zero()); L]; M];
        Self {
            frequencies,
            responses,
        }
    }

    /// Creates a new `FrequencyResponse` with logarithmically spaced frequencies.
    ///
    /// # Arguments
    /// * `freq_start` - Start frequencies for each input channel
    /// * `freq_stop` - Stop frequencies for each input channel
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with logarithmically spaced frequencies and
    ///   zero-initialized responses
    pub fn logspace(freq_start: [T; N], freq_stop: [T; N]) -> Self {
        let mut frequencies = [[T::zero(); L]; N];
        (0..N).for_each(|i| frequencies[i] = logspace(freq_start[i], freq_stop[i]));
        let responses = [[Complex::new(T::zero(), T::zero()); L]; M];
        Self {
            frequencies,
            responses,
        }
    }

    /// Creates a new `FrequencyResponse` where only one input channel has frequency data.
    ///
    /// # Arguments
    /// * `freq_start` - The start frequency for the isolated input channel
    /// * `freq_stop` - The stop frequency for the isolated input channel
    ///
    /// # Generic Arguments
    /// * `IDX` - The index of the input channel to isolate
    ///
    /// # Returns
    /// * `FrequencyResponse` instance with frequency data for the specified channel and zero-initialized responses
    pub fn isolated<const IDX: usize>(freq_start: T, freq_stop: T) -> Self {
        let mut frequencies = [[T::zero(); L]; N];
        frequencies[IDX] = logspace(freq_start, freq_stop);
        let responses = [[Complex::new(T::zero(), T::zero()); L]; M];
        Self {
            frequencies,
            responses,
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
        let frequencies = [logspace(freq_start, freq_stop); N];
        let responses = [[Complex::new(T::zero(), T::zero()); L]; M];
        Self {
            frequencies,
            responses,
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
#[derive(Copy, Clone, Debug)]
pub struct FrequencyMargin<T> {
    /// The frequency at which the phase crosses -180 degrees
    pub phase_crossover: Option<T>,
    /// The frequency at which the gain crosses 0 dB (unity gain)
    pub gain_crossover: Option<T>,
    /// The margin between the actual phase and -180 degrees at the gain crossover frequency
    pub phase_margin: Option<T>,
    /// The margin between the actual gain and 0 dB at the phase crossover frequency
    pub gain_margin: Option<T>,
}

impl<T: Float + RealField> FrequencyMargin<T> {
    /// Computes both the magnitude and phase crossover frequencies and margins of
    /// a frequency response
    ///
    /// # Arguments
    /// * `frequencies` - The array of frequency values (in rad/s)
    /// * `response` - The array of `Complex<T>` corresponding to the given frequencies
    ///
    /// # Returns
    /// * `FrequencyMargin` - the margins and crossovers of the response
    ///
    /// # Panics
    ///
    pub fn new<const L: usize>(frequencies: &[T; L], response: &[Complex<T>; L]) -> Self {
        let mut phases = [T::zero(); L];
        let mut magnitudes = [T::zero(); L];

        (0..L).for_each(|i| (magnitudes[i], phases[i]) = response[i].to_polar());

        let gain_crossover = first_crossover(frequencies, &magnitudes, T::one());
        let phase_crossover = first_crossover(frequencies, &phases, -T::pi());

        let gain_margin = phase_crossover.map(|wc| {
            // should be using frequencies.iter().position(|hz| hz == threshold, or change first_crossover to return the index)
            // unwrap is safe because the crossover frequency exists, meaning there is a corresponding
            // index in the magnitude array
            // TODO: Fix this so there doesn't need to be an unwrap
            let mag_at_crossover =
                first_crossover(&magnitudes, frequencies, wc).unwrap_or_else(|| T::zero());
            // TODO: Make a constant for this or something to avoid unwrapping
            T::from(20).unwrap_or_else(|| T::zero()) * -ComplexField::log10(mag_at_crossover)
        });

        let phase_margin = gain_crossover.map(|wc| {
            // unwrap is safe because the crossover frequency exists, meaning there is a corresponding
            // index in the phase array
            let phase_at_crossover =
                first_crossover(&phases, frequencies, wc).unwrap_or_else(|| T::zero());
            (-T::pi() - phase_at_crossover).to_degrees()
        });

        Self {
            phase_crossover,
            gain_crossover,
            phase_margin,
            gain_margin,
        }
    }
}
impl<T> Default for FrequencyMargin<T> {
    /// Creates a `FrequencyMargin` full of None
    /// # Returns
    /// * `FrequencyMargin` - with all fields set to `None`
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
/// This struct provides a matrix of `FrequencyMargin` values, where each element corresponds
/// to the stability margins for a specific input-output channel pair.
///
/// # Generic Arguments
/// - `T`: The numeric type for the stability metrics (e.g., `f32` or `f64`).
/// - `N`: The number of input channels.
/// - `M`: The number of output channels.
pub struct Margin<T, const N: usize, const M: usize>(pub [[FrequencyMargin<T>; N]; M]);

impl<T: Float + RealField, const N: usize, const M: usize> Margin<T, N, M> {
    /// Creates a new `Margin` instance from a given `FrequencyResponse`
    ///
    /// This function computes the stability margins for all input-output channel pairs in the
    /// provided `FrequencyResponse` object.
    ///
    /// # Parameters
    /// * `response` - A reference to a `FrequencyResponse` containing the frequency and response data
    ///
    /// # Generic Arguments
    /// * `L` - The number of frequency points per channel in the `FrequencyResponse`
    ///
    /// # Returns
    /// * `Margin` - A new instance containing the calculated margins for all input-output channels
    pub fn new<const L: usize>(response: &FrequencyResponse<T, L, N, M>) -> Self {
        let mut margins = [[FrequencyMargin::default(); N]; M];
        for (margin_row, res) in margins.iter_mut().zip(response.responses.iter()) {
            for (margin, frequency) in margin_row.iter_mut().zip(response.frequencies.iter()) {
                *margin = FrequencyMargin::new::<L>(frequency, res);
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
/// * `L` - Number of frequency samples provided
///
/// # Returns
/// * `Option<T>` - The crossover frequency in rad/s, or `None` if no crossover frequency is found
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
/// TODO: remove unwraps
/// # Panics
/// * if 10.0 cannot cast to T
/// * if N - 1 cannot cast to T
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
    margins: &FrequencyMargin<T>,
) where
    T: 'static + Copy + Float + From<i16> + serde::ser::Serialize,
{
    use plotly::common;
    extern crate std;
    use std::{vec::Vec, vec, format};

    let mag_db: Vec<T> = mag
        .iter()
        .map(|&m| <T as From<i16>>::from(20) * m.log10())
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
                .name(format!("Gain Margin[{row}, {col}]"))
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
            plotly::Scatter::new(vec![wc, wc], vec![<T as From<i16>>::from(-180), pm])
                .mode(common::Mode::Lines)
                .name(format!("Phase Margin[{row}, {col}]"))
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
    T: Copy + Float + RealField + From<i16> + serde::ser::Serialize,
    F: FrequencyTools<T, N, M>,
{
    use plotly::Layout;

    system.frequency_response::<L>(&mut response);
    let margins = Margin::new(&response);

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
    for (out_idx, fr) in response.responses.iter().enumerate() {
        let mut phases = [T::zero(); L];
        let mut magnitudes = [T::zero(); L];

        (0..L).for_each(|i| (magnitudes[i], phases[i]) = fr[i].to_polar());

        // render the output as the response to each input
        for (in_idx, frequency) in response.frequencies.iter().enumerate() {
            render_bode_subplot(
                &mut plot,
                frequency,
                &magnitudes,
                &phases,
                in_idx,
                out_idx,
                &margins.0[out_idx][in_idx],
            );
        }
    }

    plot
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

        assert_f32_eq!(result.unwrap(), 1.25, 1e-7); // Interpolated value
    }
    #[allow(clippy::unwrap_used)]
    #[test]
    fn multiple_threshold_instances() {
        let frequencies: [usize; 10] = core::array::from_fn(|i| i);
        let magnitudes: [usize; 10] = core::array::from_fn(|i| if i < 5 {0} else {1});
        let threshold = 1;
        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);
        assert_eq!(result.unwrap(), 5); // Interpolated value
    }
    #[allow(clippy::unwrap_used)]
    #[test]
    fn integer_interpolation() {
        let frequencies: [i32; 10] = core::array::from_fn(|i| i as i32);
        let magnitudes: [i32; 10] = core::array::from_fn(|i| if i < 5 {5} else {7});
        let threshold = 6;
        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);
        assert_eq!(result.unwrap(), 4); // Interpolated value
    }

    #[allow(clippy::unwrap_used)]
    #[test]
    fn multiple_crossovers() {
        let frequencies: [f32; 10] = core::array::from_fn(|i| i as f32);
        let magnitudes: [f32; 10] = core::array::from_fn(|i| if i % 2 == 0 {1.0} else {3.0});
        let threshold = 2.0;
        // Linear interpolation for decreasing crossover
        let result = first_crossover(&frequencies, &magnitudes, threshold);
        assert_f32_eq!(result.unwrap(), 0.5); // Interpolated value
    }
}
