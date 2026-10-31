//! Shared device conversions; magnitudes are bounded by tile and media extents.

/// Device-facing integer view of an index.
pub trait Idx {
    #[must_use]
    fn u32(self) -> u32;
}

/// Float view of index or wide-precision values.
pub trait Conv {
    #[must_use]
    fn f32(self) -> f32;
}

/// Host-width index view of a raw address.
pub trait Addr {
    #[must_use]
    fn usize(self) -> usize;
}

/// Low-order 16-bit view of packed half-precision bits.
pub trait F16Bits {
    #[must_use]
    fn u16(self) -> u16;
}

/// Signed view of an index for offset arithmetic.
pub trait Signed {
    #[must_use]
    fn isize(self) -> isize;
}

/// 64-bit address view of an index (byte offsets).
pub trait Wide {
    #[must_use]
    fn u64(self) -> u64;
}

/// Host-width view of a device scalar.
pub trait Host {
    #[must_use]
    fn usize(self) -> usize;
}

/// Truncating integer view of a float position.
pub trait Trunc {
    #[must_use]
    fn usize(self) -> usize;
}

impl Idx for usize {
    #[allow(clippy::cast_possible_truncation)]
    fn u32(self) -> u32 {
        self as u32
    }
}

impl Conv for usize {
    #[allow(clippy::cast_precision_loss)]
    fn f32(self) -> f32 {
        self as f32
    }
}

impl Conv for f64 {
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn f32(self) -> f32 {
        self as f32
    }
}

impl Addr for u64 {
    #[allow(clippy::cast_possible_truncation)]
    fn usize(self) -> usize {
        self as usize
    }
}

impl F16Bits for u32 {
    #[allow(clippy::cast_possible_truncation)]
    fn u16(self) -> u16 {
        self as u16
    }
}

impl Signed for usize {
    #[allow(clippy::cast_possible_wrap)]
    fn isize(self) -> isize {
        self as isize
    }
}

impl Wide for usize {
    fn u64(self) -> u64 {
        self as u64
    }
}

impl Host for u32 {
    fn usize(self) -> usize {
        self as usize
    }
}

impl Trunc for f32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn usize(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {

    use super::{Addr, Conv, F16Bits, Idx};

    #[test]
    fn idx_converts_and_clamps() {
        assert_eq!(7usize.u32(), 7);
        assert_eq!(usize::from(u8::MAX).u32(), 255);
        assert_eq!(1u64.usize(), 1usize);
    }

    #[test]
    fn conv_preserves_small_magnitudes() {
        assert_eq!(24usize.f32(), 24.0);
        assert_eq!(4_194_304usize.f32(), 4_194_304.0);
        assert_eq!(16_777_217usize.f32(), 16_777_216.0);
        assert_eq!(3.8f64.f32(), 3.8_f32);
    }

    #[test]
    fn f16_bits_narrow() {
        assert_eq!(0x0000_4037u32.u16(), 0x4037);
    }
}
