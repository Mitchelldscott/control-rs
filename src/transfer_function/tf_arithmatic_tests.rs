//! Test arithmetic operation for transfer function
//!
//! These rely heavily on the polynomial arithmetic operations, so check those too.

use super::*;

#[test]
fn test_add() {
    let tf = TransferFunction::new([1], [1, 1, 1]);
    let tf_plus_one = tf.add(1);
    assert_eq!(tf_plus_one.numerator.len(), 3, "Incorrect size numerator");
    assert_eq!(
        tf_plus_one.numerator,
        [2, 1, 1],
        "Incorrect numerator values"
    );
}
