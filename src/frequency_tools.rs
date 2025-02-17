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
//! phase cross specific thresholds (e.g., gain crossover at 0 dB, phase crossover at -180°).
//! - **Stability Margins:** Compute gain margins (dB) and phase margins (degrees) to assess
//! system stability.
//! - **Magnitude and Phase Response:** Evaluate transfer functions at logarithmically spaced
//! frequency points to compute magnitude and phase.
//! - **Visualization Backend:** Generate html plots using plotlyrs.
//!
//! ## Applications
//!
//! - **Control System Design:** Use frequency response metrics to tune controllers and ensure
//! desired stability margins.
//! - **Filter Analysis:** Examine the frequency characteristics of digital and analog filters.
//! - **Robustness Evaluation:** Assess how systems behave under varying conditions by analyzing
//! frequency domain behavior.
//!
//! ## References
//!
//! - Franklin, G. F., Powell, J. D., & Emami-Naeini, A. (2015). *Feedback Control of Dynamic Systems*.
//! - Dorf, R. C., & Bishop, R. H. (2016). *Modern Control Systems*.
//!
//! ### TODO:
//! - [ ] docs, docs, docs
//! - [ ] move [Margin] -> [FrequencyResponse], no point in having two structs using the same generics
//!     - add margins function to FR
//!     - add to_polar() to FR (unrelated but useful, maybe make a PolarFrequencyResponse)
//! - [ ] textbook example of trait productivity
//! - [ ] move plotly to plotly helper file (or wait for a nice gui)
//! - [ ] move FR constructors to FR factory? (configure set of FRs and target outputs without intializing anythning)

use nalgebra::{Complex, ComplexField, RealField};
use num_traits::Float;

/// Standard interface for frequency analysis tools
pub trait FrequencyTools<T, const N: usize, const M: usize>
where
    T: Float + RealField + From<i16>,
{
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

impl<T, const L: usize, const N: usize, const M: usize> FrequencyResponse<T, L, N, M>
where
    T: Float,
{
    /// Creates a new `FrequencyResponse` with specified frequency data and zeroed responses.
    ///
    /// # Arguments
    /// * `frequencies` - A 2D array containing frequency points for each input channel
    ///
    /// # Returns
    /// * `FrequencyResponse` - instance with the provided frequencies and zero-initialized responses
    pub fn default(frequencies: [[T; L]; N]) -> Self {
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
    /// * `FrequencyResponse` - instance with logarithmically spaced frequencies and zero-initialized responses
    pub fn new(freq_start: [T; N], freq_stop: [T; N]) -> Self {
        let mut frequencies = [[T::zero(); L]; N];
        (0..N).for_each(|i| frequencies[i] = generate_log_space(freq_start[i], freq_stop[i]));
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
        frequencies[IDX] = generate_log_space(freq_start, freq_stop);
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
        let frequencies = [generate_log_space(freq_start, freq_stop); N];
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

impl<T> FrequencyMargin<T> {
    /// Computes both the magnitude and phase crossover frequencies and margins of
    /// a frequency response
    ///
    /// # Arguments
    /// * `freqs` - The array of frequency values (in rad/s)
    /// * `response` - The array of `Complex<T>` corresponding to the given frequencies
    ///
    /// # Returns
    /// * `FrequencyMargin` - the margins and crossovers of the response
    pub fn new<const L: usize>(frequencies: &[T], response: &[Complex<T>]) -> Self
    where
        T: Float + RealField + From<i16>,
    {
        let mut phases = [T::zero(); L];
        let mut magnitudes = [T::zero(); L];

        (0..L).for_each(|i| (magnitudes[i], phases[i]) = response[i].to_polar());

        let gain_crossover = first_crossover::<T>(frequencies, &magnitudes, T::one(), L);
        let phase_crossover = first_crossover::<T>(frequencies, &phases, -T::pi(), L);

        let gain_margin = match phase_crossover {
            Some(wc) => {
                // should be using frequencies.iter().position(|hz| hz == threshold, or change first_crossover to return the index)
                // unwrap is safe because the crossover frequency exists, meaning there is a corresponding
                // index in the magnitude array
                let mag_at_crossover =
                    first_crossover::<T>(&magnitudes, frequencies, wc, L).unwrap();
                Some(<T as From<i16>>::from(20) * ComplexField::log10(-mag_at_crossover))
            }
            None => None,
        };

        let phase_margin = match gain_crossover {
            Some(wc) => {
                // unwrap is safe because the crossover frequency exists, meaning there is a corresponding
                // index in the phase array
                let phase_at_crossover = first_crossover::<T>(&phases, frequencies, wc, L).unwrap();
                Some((-T::pi() - phase_at_crossover).to_degrees())
            }
            None => None,
        };

        FrequencyMargin {
            phase_crossover,
            gain_crossover,
            phase_margin,
            gain_margin,
        }
    }

    /// Creates a FrequencyMargin full of None
    ///
    ///
    /// # Returns
    /// * `FrequencyMargin` - with all fields set to `None`
    pub fn default() -> Self {
        FrequencyMargin {
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

impl<T, const N: usize, const M: usize> Margin<T, N, M>
where
    T: Float + RealField + From<i16>,
{
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
        for i in 0..M {
            for j in 0..N {
                margins[i][j] =
                    FrequencyMargin::new::<L>(&response.frequencies[j], &response.responses[i]);
            }
        }

        Margin(margins)
    }
}

/// Computes the first crossover of a signal
///
/// Given two 1D arrays and a threshold this function will find the value of the first when the
/// second crosses the given threshold:
///
/// <pre>b[i] < thresh < b[i+1] || b[i] > thresh > b[i+1]</pre>
///
/// Used to find the crossover frequencies of a transfer functions magnitude and phase.
/// The magnitude crossover frequency is the frequency at which the magnitude of the transfer
/// function is 1 (0 dB). The phase crossover frequency is where the phase crosses -180°.
///
/// If no crossover is found within the given frequency range, the function returns `None` for
/// that parameter.
///
/// # Arguments
/// * `a` - The first array of values (the return value is sampled or interpolated from these)
/// * `b` - The second array of values (these are the values that may cross the threshold)
/// * `threshold` - The target threshold for the crossover (e.g., 1.0 for magnitude, -π for phase)
/// * `samples` - the number of samples in a and b
///
/// # Generic Arguments
/// * `T` - Scalar type for frequencies and values (e.g., `f32`, `f64`)
/// * `L` - Number of frequency samples provided
///
/// # Returns
/// * `Option<T>` - The crossover frequency in rad/s, or `None` if no crossover frequency is found
fn first_crossover<T>(a: &[T], b: &[T], threshold: T, samples: usize) -> Option<T>
where
    T: Float,
{
    for i in 0..samples - 1 {
        if b[i] == threshold {
            return Some(a[i]);
        }
        if (b[i] < threshold && b[i + 1] > threshold) || (b[i] > threshold && b[i + 1] < threshold)
        {
            let slope = (b[i + 1] - b[i]) / (a[i + 1] - a[i]);
            let intercept = b[i] - slope * a[i];
            return Some((threshold - intercept) / slope);
        }
    }
    // need some cleanup here
    if b[samples - 1] == threshold {
        return Some(a[samples - 1]);
    }
    None
}

/// Generate a logarithmic space array between `start` and `stop` with `n` points
pub fn generate_log_space<T: Float, const N: usize>(start: T, stop: T) -> [T; N] {
    let log_start = start.ln();
    let log_stop = stop.ln();
    let step = (log_stop - log_start) / T::from(N - 1).unwrap();
    let mut log_space = [T::zero(); N];

    for i in 0..N {
        log_space[i] = (log_start + step * T::from(i).unwrap()).exp();
    }

    log_space
}

#[cfg(feature = "std")]
/// Renders a single magnitude and phase plot on subplots
fn render_bode_subplot<T>(
    plot: &mut plotly::Plot,
    freqs: &[T],
    mag: &[T],
    phase: &[T],
    row: usize,
    col: usize,
    margins: &FrequencyMargin<T>,
) where
    T: 'static + Copy + serde::Serialize + Float + From<i16>,
{
    use plotly::common;

    let mag_db: Vec<T> = mag
        .iter()
        .map(|&m| <T as From<i16>>::from(20) * m.log10())
        .collect();
    let phase_deg: Vec<T> = phase.iter().map(|&p| p.to_degrees()).collect();

    // Add magnitude plot
    plot.add_trace(
        plotly::Scatter::new(freqs.to_vec(), mag_db)
            .mode(common::Mode::Lines)
            .name(format!("Magnitude[{row}, {col}]"))
            // .x_axis(x_axis_mag.clone())
            // .y_axis(y_axis_mag.clone()).color(Rgb::new(0, 0, 255))
            .marker(common::Marker::new()),
    );

    // Add phase plot
    plot.add_trace(
        plotly::Scatter::new(freqs.to_vec(), phase_deg)
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
                .mode(plotly::common::Mode::Lines)
                .name(format!("Phase Margin[{row}, {col}]"))
                .x_axis("x2")
                .y_axis("y2")
                .line(
                    plotly::common::Line::new()
                        .dash(plotly::common::DashType::Dot)
                        .color(plotly::color::Rgb::new(0, 0, 0)),
                ),
        );
    }
}

#[cfg(feature = "std")]
/// Renders a Bode plot for an object implementing FrequencyTools
pub fn bode<T, F, const L: usize, const N: usize, const M: usize>(
    title: &str,
    system: F,
    mut response: FrequencyResponse<T, L, N, M>,
) -> plotly::Plot
where
    T: Copy + serde::Serialize + Float + RealField + From<i16>,
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

    // extract each outputs mag/phase (make a helper in response soon)
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
    use crate::{assert_f32, assert_f64};

    #[test]
    fn test_crossover_detection() {
        // Test inputs
        let freqs: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
        let threshold = 1.0;

        // Trigger edge case where b[i] == threshold
        let result = first_crossover::<f64>(&freqs, &magnitudes, threshold, 6);

        assert!(result.is_some());
        assert_f64!(eq, result.unwrap(), 1.0); // `freqs[2]`
    }

    #[test]
    fn test_crossover_interpolation() {
        // Test inputs
        let freqs: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.9, 0.95, 1.05, 1.2, 1.3];
        let threshold = 1.0;

        // Linear interpolation for crossover
        let result = first_crossover::<f64>(&freqs, &magnitudes, threshold, 6);

        assert!(result.is_some());
        assert_f64!(eq, result.unwrap(), 1.25, 1e-10); // Interpolated value
    }

    #[test]
    fn test_no_crossover() {
        // Test inputs with no crossover
        let freqs: [f64; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f64; 6] = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99];
        let threshold = 1.0;

        // No crossover detected
        let result = first_crossover::<f64>(&freqs, &magnitudes, threshold, 6);

        assert!(result.is_none());
    }

    #[test]
    fn test_decreasing_crossover() {
        // Test inputs with decreasing crossover
        let freqs: [f32; 6] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5];
        let magnitudes: [f32; 6] = [1.2, 1.1, 1.05, 0.95, 0.8, 0.7];
        let threshold = 1.0;

        // Linear interpolation for decreasing crossover
        let result = first_crossover::<f32>(&freqs, &magnitudes, threshold, 6);

        assert!(result.is_some());
        assert_f32!(eq, result.unwrap(), 1.25, 1e-7); // Interpolated value
    }
}
