//! Conversions across display boundaries; magnitudes are bounded by media extents.

/// Display-precision float view.
pub trait Float {
    #[must_use]
    fn f32(self) -> f32;
}

impl Float for usize {
    #[allow(clippy::cast_precision_loss)]
    fn f32(self) -> f32 {
        self as f32
    }
}

/// Byte-count float view for telemetry.
pub trait Metric {
    #[must_use]
    fn f64(self) -> f64;
}

impl Metric for u64 {
    #[allow(clippy::cast_precision_loss)]
    fn f64(self) -> f64 {
        self as f64
    }
}

impl Metric for usize {
    #[allow(clippy::cast_precision_loss)]
    fn f64(self) -> f64 {
        self as f64
    }
}

/// Quantised integer views of scalar pixels.
pub trait Quant {
    #[must_use]
    fn usize(self) -> usize;
    #[must_use]
    fn u8(self) -> u8;
}

impl Quant for f32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn usize(self) -> usize {
        self as usize
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn u8(self) -> u8 {
        self as u8
    }
}

/// 8-bit channel view of a scaled float.
pub trait Narrow {
    #[must_use]
    fn u32(self) -> u32;
}

impl Narrow for f64 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn u32(self) -> u32 {
        self as u32
    }
}
