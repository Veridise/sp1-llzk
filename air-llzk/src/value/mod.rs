mod ext_felt;
mod felt;

pub use ext_felt::ExtFeltValue;
pub use felt::FeltValue;
pub use llzk_bridge::Value;

impl From<FeltValue> for Value {
    fn from(value: FeltValue) -> Self {
        value.inner
    }
}

impl From<ExtFeltValue> for Value {
    fn from(_value: ExtFeltValue) -> Self {
        unreachable!()
    }
}
