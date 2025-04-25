use sp1_stark::AirOpenedValues;

use crate::picus_codegen::{PicusVar, SignalType};
use crate::picus_codegen::Args;
use crate::PICUS_EXTRACTOR;

#[inline]
pub fn main_vars<'a>(
    n_inputs: usize,
) -> Box<dyn FnMut(usize) -> PicusVar + 'a> {
    Box::new(move |idx| {
        let mut pe = PICUS_EXTRACTOR.lock().unwrap();
        let picus_var = if idx < n_inputs {
            pe.fresh_input_picus_var()
        } else {
            let var = pe.fresh_var(SignalType::Output);
            pe.add_output(var.into());
            var
        };
        drop(pe);
        picus_var
    })
}

#[inline]
pub fn felt_arg<'a>(_arg: Args) -> Box<dyn FnMut(usize) -> PicusVar + 'a> {
    let mut pe = PICUS_EXTRACTOR.lock().unwrap();
    let fresh_var = pe.fresh_input_picus_var();
    drop(pe);
    Box::new(move |_| fresh_var)
}

#[inline]
pub fn init_vars<T, U: FnMut(usize) -> T>(count: usize, f: U) -> Vec<U::Output> {
    (0..count).map(f).collect()
}

#[inline]
pub fn air_values<T, F1: FnMut(usize) -> T, F2: FnMut(usize) -> T>(
    count: usize,
    local_fn: F1,
    next_fn: F2,
) -> AirOpenedValues<T> {
    AirOpenedValues { local: init_vars(count, local_fn), next: init_vars(count, next_fn) }
}

#[inline]
pub fn air_felt_values_from_args<'a>(
    count: usize,
    local: Args,
    next: Args,
) -> AirOpenedValues<PicusVar> {
    air_values(count, felt_arg(local), felt_arg(next))
}
