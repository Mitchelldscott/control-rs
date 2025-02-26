use super::*;

#[test]
fn constant() {
    let polynomial = Polynomial::new("x", [1]);
    assert_eq!(format!("{}", polynomial), "1");
}

#[test]
fn new() {
    let polynomial = Polynomial::new("v", [2, 1]);
    assert_eq!(format!("{}", polynomial), "2v + 1");
}

#[test]
fn neg_terms() {
    let polynomial = Polynomial::new("x'", [-2, -1]);
    assert_eq!(format!("{}", polynomial), "-2x' - 1");
}

#[test]
fn high_order() {
    let polynomial = Polynomial::new("x", [1.0; 20]);
    assert_eq!(format!("{}", polynomial), "x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1");
}

#[test]
fn empty() {
    let polynomial: Polynomial<i16, 0> = Polynomial::new("x", []);
    assert_eq!(format!("{}", polynomial), "");
}
