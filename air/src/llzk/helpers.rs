use sp1_stark::AirOpenedValues;

use super::{
    vars::{ExtFeltVar, FeltVar},
    Args, Codegen,
};

#[inline]
pub fn main_vars<'a>(
    codegen: &'a Codegen,
    arg: Args,
    n_inputs: usize,
    prefix: &'a str,
) -> Box<dyn FnMut(usize) -> FeltVar + 'a> {
    Box::new(move |idx| {
        if idx < n_inputs {
            FeltVar::ArrayArg { arg: codegen.get_func_arg(arg.clone()), idx }
        } else {
            FeltVar::Field { name: codegen.str_to_symbol(format!("{}{}", prefix, idx)) }
        }
    })
}

#[inline]
pub fn felt_arg<'a>(codegen: &'a Codegen, arg: Args) -> Box<dyn FnMut(usize) -> FeltVar + 'a> {
    Box::new(move |idx| FeltVar::ArrayArg { arg: codegen.get_func_arg(arg.clone()), idx })
}

#[inline]
pub fn felt_arg_offset<'a, const OFFSET: usize>(
    codegen: &'a Codegen,
    arg: Args,
) -> Box<dyn FnMut(usize) -> FeltVar + 'a> {
    Box::new(move |idx| FeltVar::ArrayArg {
        arg: codegen.get_func_arg(arg.clone()),
        idx: idx + OFFSET,
    })
}

#[inline]
pub fn extfelt_arg<'a>(
    codegen: &'a Codegen,
    arg: Args,
) -> Box<dyn FnMut(usize) -> ExtFeltVar + 'a> {
    Box::new(move |idx| ExtFeltVar::ArrayArg { arg: codegen.get_func_arg(arg.clone()), idx })
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
    codegen: &'a Codegen,
) -> AirOpenedValues<FeltVar> {
    air_values(count, felt_arg(codegen, local), felt_arg(codegen, next))
}

#[inline]
pub fn air_extfelt_values_from_args<'a>(
    count: usize,
    local: Args,
    next: Args,
    codegen: &'a Codegen,
) -> AirOpenedValues<ExtFeltVar> {
    air_values(count, extfelt_arg(codegen, local), extfelt_arg(codegen, next))
}
