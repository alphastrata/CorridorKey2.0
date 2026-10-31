//! Device-facing integer view of an index; magnitudes are bounded by tensor extents.

pub trait Idx {
    #[must_use]
    fn u32(self) -> u32;
}

impl Idx for usize {
    #[allow(clippy::cast_possible_truncation)]
    fn u32(self) -> u32 {
        self as u32
    }
}
/// Display-precision float view.
pub trait Conv {
    #[must_use]
    fn f32(self) -> f32;
}

impl Conv for usize {
    #[allow(clippy::cast_precision_loss)]
    fn f32(self) -> f32 {
        self as f32
    }
}

impl Conv for u32 {
    #[allow(clippy::cast_precision_loss)]
    fn f32(self) -> f32 {
        self as f32
    }
}

/// Byte-count float view.
pub trait Metric {
    #[must_use]
    fn f64(self) -> f64;
}

impl Metric for usize {
    #[allow(clippy::cast_precision_loss)]
    fn f64(self) -> f64 {
        self as f64
    }
}

/// Host-index and precision-capped views of arithmetic results.
pub trait Narrow {
    #[must_use]
    fn usize(self) -> usize;
    #[must_use]
    fn f32(self) -> f32;
}

impl Narrow for f64 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn usize(self) -> usize {
        self as usize
    }

    #[allow(clippy::cast_possible_truncation)]
    fn f32(self) -> f32 {
        self as f32
    }
}
