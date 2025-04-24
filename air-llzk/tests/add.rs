mod chips;
mod common;

use air_llzk::OutputFormats;

#[test]
pub fn test_mlir() {
    common::test_chip::<chips::add::AddChip>(OutputFormats::Assembly);
}

#[test]
pub fn test_picus() {
    common::test_chip::<chips::add::AddChip>(OutputFormats::Picus);
}
