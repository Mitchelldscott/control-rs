# Control-rs

Control-rs is a numerical modeling and control system library designed for embedded applications. Inspired by MATLAB’s 
control systems toolbox, this crate provides a structured approach to system modeling, analysis, and numerical design 
while maintaining a lightweight footprint suitable for real-time and resource-constrained environments. 

## Features

* **Modeling**: Support for Polynomial, Transfer Function, State-Space, and other nonlinear representations
* **Analysis**: Tools for classical, modern and robust system analysis
* **Synthesis**: Direct and data-driven methods to create models
* **Simulation**: Easy model integration and data vizualization

## Installation (Not Supported... haven't published the crate yet, clone this repo instead)

Add this to your `Cargo.toml`:

```toml
[dependencies]
control-rs = "0.1.0"
```

or run

```bash
cargo add control-rs
```

## Usage

Here's a simple example to get you started:

```rust
use control_rs::transfer_function::{TransferFunction, dcgain};

fn main() {
    let mut tf = Transferfunction::new([1.0], [1.0, 0.0]);
    println!("DC Gain of TF: {}", dcgain(tf));
}
```

```rust
use control_rs::{StateSpace, TransferFunction, transfer_function::as_monic};

fn main() {
    let tf = TransferFunction::new([2.0, 4.0], [1.0, 1.0, 4.0, 0.0, 0.0]);
    let (num, den) = as_monic(&tf);

    assert_eq!(
        den[0], 1.0,
        "Transfer Function denominator is not monic\n{tf}"
    );

    let ss: StateSpace<_, 4, 1, 1> = control_canonical(num, den);
}
```

## Examples

Examples are either based on a textbook problem or demo a practical application. Future work includes providing (a) cargo template(s) for hardware-integrated examples (with specific sensors and mcu's).

## Testing

### Unit Tests

* Located within the module containing the unit being tested (e.g., `src/<module>/...`).
* Simple tests may reside in mod.rs, while more complex tests will have dedicated files to keep mod.rs concise.
* Each test case should be:
  * Short
  * Independent
  * Clearly descriptive of the test objective

### Integration Tests

* Located in the tests/ directory.
* Designed to verify high-level functionality of the tools.
* Example: Confirm that a simple model maintains consistency across different model formats.

## Documentation

The documentation provides theoretical references and specific user guidance. Each module should include:

* A concise conceptual description (similar to [MathWorks TransferFunction docs](https://www.mathworks.com/help/control/ug/transfer-functions.html))
* Links to theoretical references
* Use cases (verified with doctests)

## Book

In the future it would be great to have a book that walks through a series of smaller projects to accomplish a larger one. For example designing analytical models for a few sensors and actuators and integrating them to design an estimator and controller.

## Contributing

We welcome contributions!

## License

This project should have a license, but vscode kept complaining. I'll bring it back soon.

Thank you for using Control-rs!
