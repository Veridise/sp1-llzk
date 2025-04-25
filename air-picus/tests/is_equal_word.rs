mod chips;
mod common;

#[test]
pub fn test_picus() {
    common::test_chip::<chips::is_equal_word::Chip>();
}
