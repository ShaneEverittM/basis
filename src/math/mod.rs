pub fn add(x: f64, y: f64) -> f64 { x + y }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_test() {
        assert!((3f64 - add(1f64, 2f64)).abs() < f64::EPSILON)
    }
}