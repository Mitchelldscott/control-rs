use std::mem::MaybeUninit;

/// A trait for storage types that manage a 1D, statically known, fixed-size buffer.
///
/// This trait provides a low-level abstraction for a contiguous block of memory
/// containing exactly `N` elements of type `T`. It is intended as a foundational
/// building block for more complex data structures, such as multidimensional dense
/// arrays or sparse data structures.
///
/// Implementors of this trait are responsible for managing the memory for these N
/// elements. Operations that might fail due to dynamic conditions (like out-of-bounds
/// access on a raw index) are typically marked `unsafe`, requiring the caller to
/// guarantee preconditions.
///
///
/// ## Multi-Dimensional Structures
///
/// `StaticStorage` itself is 1D-centric. Higher-level structures (e.g., a
/// `DenseMatrix<T, const ROWS: usize, const COLS: usize, S: StaticStorage<T, {ROWS * COLS}>>`)
/// would *use* an implementor of `StaticStorage` for their backing memory. These
/// higher-level types are responsible for mapping multidimensional indices to the
/// flat `usize` index required by this trait's methods like `get_unchecked`.
///
/// For sparse structures, multiple instances of `StaticStorage` might be used to store
/// different components (e.g., non-zero values, indices, pointers), each with its
/// own `N` representing the capacity of that specific component array.
///
pub trait StaticStorage<T, const N: usize>: Sized {
    type Index;

    /// Creates a new storage by consuming elements from the provided iterator.
    ///
    /// It will attempt to fill all N slots of the storage.
    /// - If the `iterator` yields fewer than `N` items, the remaining slots
    ///   are filled by cloning the `default` value.
    /// - If the `iterator` yields more than `N` items, only the first `N` items
    ///   are used, and the rest are ignored (truncated).
    ///
    /// This function does not error or panic.
    ///
    /// # Generic Arguments
    /// - `I`: An iterator that produces items of type `T`.
    ///
    /// # Arguments
    /// - `iterator`: The iterator to source elements from.
    /// - `default`: A reference to the value used for padding if the iterator is too short.
    ///              Requires `T: Clone`.
    fn from_iterator_with_default<I: IntoIterator<Item = T>>(iterator: I, default: T) -> Self;

    /// Returns a slice providing a view into all `N` elements of the storage.
    fn as_slice(&self) -> &[T];

    /// Compute the linear index corresponding to the specified index type
    fn linear_index(&self, index: Self::Index) -> usize;

    /// Gets the address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked_linear(&self, i: usize) -> *const T {
        self.ptr().wrapping_add(i)
    }

    /// Gets the address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked(&self, index: Self::Index) -> *const T {
        self.get_address_unchecked_linear(self.linear_index(index))
    }

    /// Retrieves a reference to the i-th element without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn get_unchecked_linear(&self, i: usize) -> &T {
        unsafe { &*self.get_address_unchecked_linear(i) }
    }

    /// Retrieves a reference to the i-th element without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn get_unchecked(&self, index: Self::Index) -> &T {
        unsafe { self.get_unchecked_linear(self.linear_index(index)) }
    }

    /// Returns the compile-time capacity (number of elements) of the storage.
    ///
    /// This will always be equal to `N`.
    #[must_use]
    #[inline(always)]
    fn capacity(&self) -> usize {
        N
    }

    /// Returns a raw pointer to the first element of the storage.
    ///
    /// If `N` is 0, the returned pointer might not be dereferenceable but
    /// should be properly aligned for `T` (e.g., like `core::ptr::NonNull::dangling()`).
    ///
    /// # Safety
    /// The caller must ensure that the storage outlives the pointer and that
    /// any access through this pointer is valid (e.g., within bounds `0: N`).
    /// Dereferencing the pointer is only safe if `N > 0`.
    fn ptr(&self) -> *const T;
}

pub trait StaticStorageMut<T, const N: usize>: StaticStorage<T, N> {
    /// Returns a mutable slice providing a view into all `N` elements of the storage.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Returns a mutable raw pointer to the first element of the storage.
    ///
    /// If `N` is 0, the returned pointer might not be dereferenceable but
    /// should be properly aligned for `T`.
    ///
    /// # Safety
    /// The caller must ensure that the storage outlives the pointer,
    /// that any access through this pointer is valid, and that Rust's
    /// aliasing rules are upheld (no other active references when modifying data).
    /// Dereferencing the pointer for writing is only safe if `N > 0`.
    fn ptr_mut(&mut self) -> *mut T;

    /// Gets the mutable address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked_linear_mut(&mut self, i: usize) -> *mut T {
        self.ptr_mut().wrapping_add(i)
    }

    /// Gets the mutable address of the i-th matrix component without performing bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, dereferencing the result will cause undefined behavior.
    #[inline]
    fn get_address_unchecked_mut(&mut self, index: Self::Index) -> *mut T {
        self.get_address_unchecked_linear_mut(self.linear_index(index))
    }

    /// Retrieves a mutable reference to the i-th element without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    unsafe fn get_unchecked_linear_mut(&mut self, i: usize) -> &mut T {
        unsafe { &mut *self.get_address_unchecked_linear_mut(i) }
    }

    /// Retrieves a mutable reference to the element at `index` without bound-checking.
    ///
    /// # Safety
    /// If the index is out of bounds, the method will cause undefined behavior.
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, index: Self::Index) -> &mut T {
        unsafe { &mut *self.get_address_unchecked_mut(index) }
    }
}

pub struct DenseArray<T, const N: usize>([T; N]);

impl<T: Clone, const N: usize> StaticStorage<T, N> for DenseArray<T, N> {
    type Index = usize;

    fn from_iterator_with_default<I: IntoIterator<Item = T>>(iterator: I, default: T) -> Self {
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
            DenseArray((uninit_array.as_ptr() as *const [T; N]).read())
        }
    }

    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    fn linear_index(&self, index: Self::Index) -> usize {
        index
    }

    fn ptr(&self) -> *const T {
        self.0.as_ptr()
    }
}

impl<T: Clone, const N: usize> StaticStorageMut<T, N> for DenseArray<T, N> {
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }

    fn ptr_mut(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }
}

pub struct SparseArray<T, const N: usize>([(usize, T); N]);

impl<T: Clone, const N: usize> StaticStorage<(usize, T), N> for SparseArray<T, N> {
    type Index = usize;

    fn from_iterator_with_default<I: IntoIterator<Item = (usize, T)>>(
        iterator: I,
        default: (usize, T),
    ) -> Self {
        // SAFETY: [MaybeUninit<T>; N] is valid.
        let mut uninit_array: [MaybeUninit<(usize, T)>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };
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
            SparseArray((uninit_array.as_ptr() as *const [(usize, T); N]).read())
        }
    }

    fn as_slice(&self) -> &[(usize, T)] {
        self.0.as_slice()
    }

    fn linear_index(&self, index: Self::Index) -> usize {
        for (j, (i, _)) in self.as_slice().iter().enumerate() {
            if *i == index {
                return j;
            }
        }

        usize::MAX
    }

    fn ptr(&self) -> *const (usize, T) {
        self.0.as_ptr()
    }
}

impl<T: Clone, const N: usize> StaticStorageMut<(usize, T), N> for SparseArray<T, N> {
    fn as_mut_slice(&mut self) -> &mut [(usize, T)] {
        self.0.as_mut_slice()
    }

    fn ptr_mut(&mut self) -> *mut (usize, T) {
        self.0.as_mut_ptr()
    }
}

fn main() {}
