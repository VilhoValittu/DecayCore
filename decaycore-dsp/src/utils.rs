// Utility functions for safe float arithmetic

#[inline]
pub fn safe_f64(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

#[inline]
pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

// Linear interpolation between two points
#[inline]
pub fn linear_interp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if (x1 - x0).abs() < 1e-14 {
        return (y0 + y1) * 0.5;
    }
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}
