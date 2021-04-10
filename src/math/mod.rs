pub fn add(x: u64, y: u64) -> u64 {
    x + y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_test() {
        assert_eq!(3, add(1, 2))
    }
}