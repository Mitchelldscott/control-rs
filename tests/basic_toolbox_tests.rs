#[cfg(test)]
mod basic_model_tests {

    use control_rs::{
        integrators::runge_kutta4,
        state_space::{
            utils::{control_canonical, zoh},
            StateSpace,
        },
        systems::{DynamicalSystem, NLModel},
    };
    use nalgebra::{Matrix2x1, Vector2};

    type ScalarType = f32;

    type ParticleState = Matrix2x1<ScalarType>;
    type ParticleInput = ScalarType;
    type ParticleOutput = ScalarType;

    /// 1D particle model with position and velocity states
    struct Particle1D {
        state: ParticleState,
        control: ParticleInput,
    }

    impl DynamicalSystem<ParticleInput, ParticleState, ParticleOutput> for Particle1D {
        fn dynamics(&self, state: ParticleState, input: ParticleInput) -> ParticleState {
            ParticleState::new(state[1], input - 0.1 * state[1])
        }

        fn output(&self, state: ParticleState, _input: ParticleInput) -> ParticleOutput {
            state[0]
        }
    }
    use nalgebra::SMatrix;
    impl
        NLModel<
            ParticleInput,
            ParticleState,
            ParticleOutput,
            SMatrix<ScalarType, 2, 2>,
            SMatrix<ScalarType, 2, 1>,
            SMatrix<ScalarType, 1, 2>,
            SMatrix<ScalarType, 1, 1>,
        > for Particle1D
    {
        fn linearize(
            &self,
            _state: ParticleState,
            _input: ParticleInput,
        ) -> StateSpace<
            SMatrix<ScalarType, 2, 2>,
            SMatrix<ScalarType, 2, 1>,
            SMatrix<ScalarType, 1, 2>,
            SMatrix<ScalarType, 1, 1>,
        > {
            control_canonical([1.0], [0.1, 0.0])
        }
    }

    #[test]
    fn particle_simulation_comparison() {
        let dt = 0.1;
        let control = 0.2;
        let int_steps = 100;

        // Create a new particle model
        let model = Particle1D {
            state: Vector2::zeros(),
            control,
        };

        // linearize the model
        let ss = model.linearize(model.state, model.control);

        // discretize the linear model with a zero-order-hold
        let ssd = zoh(&ss, dt);

        let mut x = model.state;
        let mut xss = model.state;
        let mut xssd = model.state;

        // simulate for 100 steps
        for i in 0..100 {
            // NL
            x = runge_kutta4(&model, x, control, 0.0, dt, dt / int_steps as ScalarType);

            // Continuous State-Space
            for _ in 0..int_steps {
                xss += ss.dynamics(xss, control) * dt / int_steps as ScalarType;
            }

            assert!(
                (x - xss).norm() < 5e-4,
                "nonlinear vs continuous simulation drift = {} after {} iterations",
                i + 1,
                (x - xss).norm(),
            );

            // Discrete State-Space
            xssd = ssd.dynamics(xssd, control);

            assert!(
                (x - xssd).norm() < 6e-5,
                "nonlinear vs discrete simulation drift = {} after {} iterations",
                i + 1,
                (x - xssd).norm(),
            )
        }
    }
}

#[cfg(test)]
mod tf_frequency_tool_tests {
    use control_rs::{assert_f64, frequency_tools::*, transfer_function::TransferFunction};

    /// Test gain and phase margins.
    #[test]
    fn test_margins() {
        let tf = TransferFunction::new([10.0], [1.0, -5.0]);
        // Frequencies to evaluate (in rad/s)
        let frequencies = [0.0, 0.1, 1.0, 10.0];

        // Call the bode function
        let mut response = FrequencyResponse::default([frequencies]);
        tf.frequency_response(&mut response);

        let margins = Margin::new(&response);

        if let FrequencyMargin {
            phase_crossover: Some(phase_crossover),
            gain_crossover: Some(gain_crossover),
            phase_margin: Some(phase_margin),
            gain_margin: Some(gain_margin),
        } = margins.0[0][0]
        {
            assert_f64!(eq, phase_crossover, 0.0, 0.1);
            assert_f64!(eq, gain_crossover, 8.66, 0.1);
            assert_f64!(eq, phase_margin, 60.0, 2.5); // wide error range because crossover is imprecise with so few points.
            assert_f64!(eq, gain_margin, -6.02, 0.01);
        } else {
            panic!(
                "Failed to compute gain and phase margins {:?}",
                margins.0[0][0]
            );
        }
    }
}

#[cfg(test)]
#[cfg(feature = "std")]
mod bode_and_nyquist_plot_tests {
    use control_rs::{
        frequency_tools::{bode, FrequencyResponse},
        transfer_function::TransferFunction,
    };

    #[test]
    fn bode_plot() {
        // Define a test transfer function
        let title = "Demo Bode Plot";
        let tf = TransferFunction::new([1.0], [1.0, 1.0, 1.0]);

        let response = FrequencyResponse::<f64, 100, 1, 1>::new([0.1], [10.0]);

        std::fs::create_dir_all("../target/plots").unwrap();
        bode(title, tf, response).write_html("../target/plots/test_bode_plot.html");
    }

    #[test]
    fn nyquist_plot() {
        // Define a test transfer function
        let _title = "Demo Nyquist Plot";
        let _tf = TransferFunction::new([1.0], [1.0, 1.0, 1.0]);

        let _response = FrequencyResponse::<f64, 100, 1, 1>::new([0.1], [10.0]);

        // println!("{title}\n{tf}\n{:?}", response.responses);
    }
}
