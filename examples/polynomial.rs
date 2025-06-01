// #![no_std]

use core::{
    array,
    fmt,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg},
    slice
};

use num_traits::{One, Zero};

use nalgebra::{Const, DimDiff, DimSub, U1};

#[inline]
fn from_iterator_with_default<I, T, const N: usize>(
    iterator: I,
    default: &T,
) -> [T; N]
where
    T: Clone,
    I: IntoIterator<Item = T>,
{
    // SAFETY: [MaybeUninit<T>; N] is valid.
    let mut uninit_array: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut count = 0;

    // SAFETY: zip() will only iterate over N.min(iter.len()) elements so no out-of-bounds access
    // can occur.
    for (c, d) in uninit_array.iter_mut().zip(iterator.into_iter()) {
        *c = MaybeUninit::new(d);
        count += 1;
    }

    // SAFETY: T: Clone, and we are initializing all remaining uninitialized slots.
    for c in uninit_array.iter_mut().skip(count) {
        *c = MaybeUninit::new(default.clone());
    }

    // SAFETY:
    // - All N elements of uninit_array have now been initialized.
    // - MaybeUninit<T> does not drop its content, so no double-drop will occur.
    // - We can safely transmute it to [T; N] by reading the pointer.
    unsafe {
        // Get a pointer to the uninit_array array.
        // Cast it to a pointer to an array of T.
        // Then read() the value from that pointer.
        // This is equivalent to transmute from [MaybeUninit<T>; N] to [T; N].
        (uninit_array.as_ptr() as *const [T; N]).read()
    }
}

pub struct Polynomial<T, const N: usize> {
    coefficients: [T; N],
}

impl<T, const N: usize> From<[T; N]> for Polynomial<T, N> {
    #[inline]
    fn from(coefficients: [T; N]) -> Self {
        Self { coefficients }
    }
}

impl<T, const N: usize> Polynomial<T, N> {
    #[inline]
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self::from(array::from_fn(cb))
    }

    #[inline]
    #[must_use]
    unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.coefficients.get_unchecked(index)
    }

    #[inline]
    #[must_use]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.coefficients.get_unchecked_mut(index)
    }

    /// Access the coefficients as a slice iter
    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.coefficients.iter()
    }

    /// Access the coefficients as a mutable slice iter
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.coefficients.iter_mut()
    }
}

impl<T: Clone, const N: usize> Polynomial<T, N> {
    #[inline]
    pub fn from_iterator_with_default<I: IntoIterator<Item=T>>(iterator: I, default: &T) -> Self {
        from_iterator_with_default(iterator, default).into()
    }
    #[inline]
    pub fn resize<const M: usize>(self, default: &T) -> Polynomial<T, M>{
        Polynomial::<T, M>::from_iterator_with_default(self.coefficients, default)
    }
}

impl<T: Zero, const N: usize> Polynomial<T, N> {
    #[inline]
    pub fn degree(&self) -> Option<usize> {
        for (i, a_i) in self.coefficients.iter().enumerate().rev() {
            if !a_i.is_zero() { return Some(i); }
        }
        None
    }
}

pub enum PolynomialDerivative<T, const N: usize> {
    Zero(T),
    Ok(Polynomial<T, N>),
}

pub enum PolynomialIntegral<T, const N: usize> {
    Constant(T),
    Ok(Polynomial<T, N>),
    Truncated(Polynomial<T, N>),
}

impl<T: Clone + Zero + One + Add<Output = T> + Mul<Output = T>, const N: usize> Polynomial<T, N> {
    #[inline]
    fn derivative_internal(&self) -> Self {
        // Polynomial::from_iterator_with_default(
        //     self.iter().skip(1).enumerate().map(|(i, a_i)|
        //         (0..i).fold(a_i.clone(), |result, _| result + a_i.clone())
        //     ),
        //     &T::zero()
        // )
        let mut exponent = T::zero();
        Polynomial::from_iterator_with_default(
            self.iter().skip(1).map(|a_i| {
                exponent = exponent.clone() + T::one();
                a_i.clone() * exponent.clone()
            }),
            &T::zero()
        )
    }

    #[inline]
    pub fn derivative(&self) -> PolynomialDerivative<T, N> {
        if let Some(degree) = self.degree() {
            PolynomialDerivative::Ok(self.derivative_internal())
        } else {
            PolynomialDerivative::Zero(T::zero())
        }
    }
}

impl<T: Clone + Zero + One + Add<Output = T> + Div<Output = T>, const N: usize> Polynomial<T, N> {
    #[inline]
    fn integral_internal(&self, constant: T) -> Self {
        let mut exponent = T::one();
        Polynomial::from_iterator_with_default(
            [constant].into_iter().chain(
                self.iter().enumerate().map(|(_, a_i)| {
                    exponent = exponent.clone() + T::one();
                    a_i.clone() / exponent.clone()
                })
            ),
            &T::zero()
        )
    }
    #[inline]
    pub fn integral(&self, constant: T) -> PolynomialIntegral<T, N> {
        if let Some(degree) = self.degree() {
            let integral = self.integral_internal(constant);
            if degree + 1 == N { PolynomialIntegral::Truncated(integral) }
            else { PolynomialIntegral::Ok(integral) }
        }
        else { PolynomialIntegral::Constant(constant) }
    }
}

impl<T, const N: usize> fmt::Display for Polynomial<T, N>
where
    T: Clone + Zero + One + PartialOrd + Neg<Output = T> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut n = N;
        for (i, a_i) in self.iter().enumerate().rev() {
            if a_i.is_zero() {
                n = n.saturating_sub(1);
                continue;
            }
            if n > 1 && n > i + 1 {
                write!(f, " {} ", if *a_i >= T::zero() { "+" } else { "-" })?;
            } else if *a_i < T::zero() {
                write!(f, "-")?;
            }
            let abs_a_i = if *a_i < T::zero() {
                a_i.clone().neg()
            } else {
                a_i.clone()
            };
            if !abs_a_i.is_one() || i == 0 {
                write!(f, "{}", abs_a_i)?;
            }
            if i > 0 {
                write!(f, "x")?;
                if i > 1 {
                    write!(f, "^{}", i)?;
                }
            }
        }
        Ok(())
    }
}

pub struct Equation<T, N, S> {
    data: S,
    _phantoms: PhantomData<(T, N)>,
}

pub struct Roots<T, N, S> {
    roots: S,
    _phantoms: PhantomData<(T, N)>,
}

use nalgebra::{DefaultAllocator, allocator::Allocator, Dim, Scalar, Owned, DimAdd};
impl<T: Scalar + Zero, D: Dim> Roots<T, D, Owned<T, D, U1>>
where
    D: DimAdd<U1>,
    DefaultAllocator: Allocator<D, U1>,
{
    fn roots<const N: usize>(degree: D, p: Polynomial<T, N>) -> Self 
    where
        Const<N>: DimSub<U1, Output = D>
    {
        Self {
            roots: DefaultAllocator::allocate_from_iterator(degree, U1, p.coefficients.into_iter()),
            _phantoms: PhantomData,
        }
    }
}

fn main() {
    let p = Polynomial::from([1, 2, 3]);
    if let PolynomialDerivative::Ok(dp) = p.derivative() {
        println!("{p} -> {dp} |");
        if let PolynomialDerivative::Ok(ddp) = dp.derivative() {
            println!("{p} -> {dp} -> {ddp} |");
            if let PolynomialDerivative::Ok(dddp) = ddp.derivative() {
                println!("{p} -> {dp} -> {ddp} -> {dddp} |");
                if let PolynomialDerivative::Zero(ddddp) = dddp.derivative() {
                    println!("{p} -> {dp} -> {ddp} -> {dddp} -> {ddddp} |");
                }
            }
        }
    }
    
    let roots = Roots::roots(Const::<2>, p);
    println!("{:?}", roots.roots);
}