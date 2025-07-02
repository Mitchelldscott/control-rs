# Documentation Standards for `control-rs`

These rules establish a rigorous and consistent documentation standard for a Rust native control systems toolbox 
intended for safety-critical applications. They integrate best practices from the official Rust API Guidelines with the 
stringent documentation requirements of safety standards like DO-178C and ISO 26262.

# 1. Crate level documentation

Every crate must provide comprehensive crate-level documentation at the root of the crate. This documentation serves as 
the primary technical overview and safety manual for the crate.
* Description: A clear and concise description of the crate's purpose and its role within the control system.
* Safety-Criticality: A mandatory section specifying the Design Assurance Level (DAL) or Automotive Safety Integrity 
Level (ASIL) the crate is designed to meet.
* Core Concepts: An explanation of the fundamental principles and algorithms implemented in the crate.
* Usage: A practical example demonstrating the crate's primary use case. This example must be runnable with cargo test.
* Features: A list and explanation of all cargo features, especially those that alter functionality or introduce 
dependencies.
* Limitations: A clear statement of any known limitations, assumptions, or operational constraints.

# 2. Public API Documentation

All public items (modules, structs, enums, functions, traits, and macros) must be thoroughly documented using ///. The 
documentation for each item must follow a consistent and explicit structure.

## 2.1. General Structure

The documentation for every public item should adhere to the following order:
* **Summary**: A brief, one-sentence description of the item's purpose.
* **Detailed Description**: A more in-depth explanation of the item's functionality and its intended use.
* **Generic Arguments**: Generic fields that can be passed to the function or type.
* **Arguments**: Arguments that a function expects.
* **Returns**: For functions, a clear description of the return value and its meaning in all conditions.
* **Errors**: A clear explanation of all possible Err variants returned by the item.
* **Safety**: A mandatory section detailing all safety-related aspects.
* **Panics**: An explicit list of all conditions under which the item will panic.
* **Example**: At least one runnable example that demonstrates typical usage.

### Example of function docs
```rust
/// Finds the largest index of a non-zero value in a slice.
///
/// This function iterates through a slice in reverse and checks if each value `.is_zero()`. It
/// will return after the first time the condition is **false**.
///
/// # Generic Arguments
/// * `T` - field type of the array, which must implement [Zero].
///
/// # Arguments
/// * `coefficients` - a slice of `T`.
///
/// # Returns
/// * `Option<usize>`
///     * `Some(index)` - The largest index containing a non-zero value.
///     * `None` - If the slice is empty or all elements are zero.
/// 
/// # Panics
/// This function does not panic.
/// 
/// # Safety 
/// This function does not use `unsafe` code. The logical correctness of the result is dependent on the correctness of 
/// the `.is_zero()` implementation for type `T`.
/// 
/// # Example
/// ```
/// use control_rs::polynomial::utils::largest_nonzero_index;
/// assert_eq!(largest_nonzero_index::<u8>(&[]), None);
/// assert_eq!(largest_nonzero_index(&[0, 1]), Some(1));
/// assert_eq!(largest_nonzero_index(&[1, 0]), Some(0));
/// assert_eq!(largest_nonzero_index(&[0, 0]), None);
/// assert_eq!(largest_nonzero_index(&[1]), Some(0));
/// ```
```

## 2.2. The #[safety] Section: A Contract for Critical Code

For any function, method, or unsafe block that has safety implications, a dedicated #[safety] section is mandatory. 
This section must explicitly detail the contract the caller must uphold to ensure safe execution.

The #[safety] documentation must clearly state:
* Pre-conditions: The conditions that must be true before calling the function. This includes, but is not limited to, 
the state of hardware, the validity of inputs, and the expected configuration of the system.
* Post-conditions: The state of the system after the function has executed successfully. This describes the expected 
outputs and any side effects.
* Invariants: The properties that are guaranteed to remain unchanged during the execution of the function.
* Assumptions: Any assumptions made about the environment or other parts of the system.
* Rationale for unsafe: For any unsafe block, a clear justification for its use and a detailed explanation of how its 
safety is guaranteed by the surrounding safe code.

### Example of #[safety] documentation:
```rust
/// # Safety
///
/// This function directly manipulates hardware registers and must be used with extreme care.
/// The caller MUST ensure the following conditions are met:
/// - The system clock for the peripheral has been enabled.
/// - The provided `base_address` is a valid memory-mapped address for the UART peripheral.
/// - No other part of the system is concurrently accessing this UART peripheral.
///
/// Failure to adhere to these conditions will result in undefined behavior.
```

# 3. Examples and Testing
* All examples must be runnable via cargo test.
* Examples should use the ? operator for error handling and avoid unwrap(), expect(), or panic!() unless demonstrating 
a panic condition.
* Examples for safety-critical functions should demonstrate both correct usage and, where possible, how incorrect usage 
is handled.

# 4. Traceability

To comply with safety standards, documentation must facilitate traceability between requirements, design, code, and 
tests.
* Requirement Tags: Where applicable, include tags in the documentation to link code to specific software requirements. 
* Design Rationale: For complex or critical algorithms, include a brief explanation of the design choices and 
references to relevant design documents.

By adhering to these rigorous documentation rules, the Rust native control systems toolbox will provide the necessary 
clarity, consistency, and explicit safety information required for use in safety-critical applications.

# Resources
* [Meet safe and unsafe](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)
* [unsafe keyword](https://doc.rust-lang.org/std/keyword.unsafe.html)
* [Unsafe Rust](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)
* [Awesome safety critical](https://awesome-safety-critical.readthedocs.io/en/latest/)
* [Rustdoc book](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html)