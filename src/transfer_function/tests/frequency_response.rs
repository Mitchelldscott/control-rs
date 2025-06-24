#[cfg(test)]
mod frequency_tools_tests {
    use crate::TransferFunction;
    use crate::{
        assert_f64_eq,
        frequency_tools::{FrequencyResponse, FrequencyTools},
    };

    #[test]
    fn test_low_pass_mag_phase() {
        // Define a simple first-order low-pass filter: H(s) = 1 / (s + 1)
        let tf = TransferFunction::new([1.0], [1.0, 1.0]);

        // Frequencies to evaluate (in rad/s)
        let frequencies = [
            0.1,  // Below cutoff frequency
            1.0,  // At cutoff frequency
            10.0, // Above cutoff frequency
        ];

        // Call the bode function
        let mut response = FrequencyResponse::new([frequencies]);
        tf.frequency_response(&mut response);

        let (magnitudes, phases) = response.mag_phase(0);

        // Expected results
        let expected_magnitudes = [
            0.995_037_190_209_989_2,          // |H(j0.1)| ≈ 1 / sqrt(1 + (0.1)^2)
            core::f64::consts::FRAC_1_SQRT_2, // |H(j1)| ≈ 1 / sqrt(1 + (1)^2)
            0.099_503_719_020_998_92,         // |H(j10)| ≈ 1 / sqrt(1 + (10)^2)
        ];
        let expected_phases = [
            -0.099_668_652_491_162_04,     // arg(H(j0.1)) ≈ -atan(0.1)
            -core::f64::consts::FRAC_PI_4, // arg(H(j1)) ≈ -atan(1)
            -1.471_127_674_303_734_7,      // arg(H(j10)) ≈ -atan(10)
        ];

        // Validate results
        for i in 0..frequencies.len() {
            assert_f64_eq!(magnitudes[i], expected_magnitudes[i]);
            assert_f64_eq!(phases[i], expected_phases[i]);
        }
    }
}
