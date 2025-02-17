
#[cfg(test)]
mod frequency_tools_tests {
    use crate::TransferFunction;
    use crate::{assert_f64, frequency_tools::FrequencyResponse};
    
    #[test]
    fn test_low_pass_mag_phase() {
        // Define a simple first-order low-pass filter: H(s) = 1 / (s + 1)
        let tf = TransferFunction::new([1.0], [1.0, 1.0]);

        // Frequencies to evaluate (in rad/s)
        let freqs = [
            0.1,  // Below cutoff frequency
            1.0,  // At cutoff frequency
            10.0, // Above cutoff frequency
        ];

        // Call the bode function
        let mut response = FrequencyResponse::default([freqs]);
        tf.frequency_response::<3>(&mut response);

        let mut phases = [f64::zero(); 3];
        let mut magnitudes = [f64::zero(); 3];

        (0..3).for_each(|i| (magnitudes[i], phases[i]) = response.responses[0][i].to_polar());

        // Expected results
        let expected_magnitudes = [
            0.9950371902099892,  // |H(j0.1)| ≈ 1 / sqrt(1 + (0.1)^2)
            0.7071067811865476,  // |H(j1)| ≈ 1 / sqrt(1 + (1)^2)
            0.09950371902099892, // |H(j10)| ≈ 1 / sqrt(1 + (10)^2)
        ];
        let expected_phases = [
            -0.09966865249116204, // arg(H(j0.1)) ≈ -atan(0.1)
            -0.7853981633974483,  // arg(H(j1)) ≈ -atan(1)
            -1.4711276743037347,  // arg(H(j10)) ≈ -atan(10)
        ];

        // Validate results
        for i in 0..freqs.len() {
            assert_f64!(eq, magnitudes[i], expected_magnitudes[i]);
            assert_f64!(eq, phases[i], expected_phases[i]);
        }
    }
}