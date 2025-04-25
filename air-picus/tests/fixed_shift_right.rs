mod chips;
mod common;


#[test]
pub fn test_picus() {
    common::test_chip::<chips::fixed_shift_right::Chip>();
}