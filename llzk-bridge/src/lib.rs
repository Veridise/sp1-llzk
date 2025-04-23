#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::CStr;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl Send for Value {}
unsafe impl Send for Symbol {}
unsafe impl Sync for Value {}
unsafe impl Sync for Symbol {}

impl From<&CStr> for MlirStringRef {
    fn from(str: &CStr) -> Self {
        unsafe { mlirStringRefCreateFromCString(str.as_ptr()) }
    }
}

impl From<&str> for MlirStringRef {
    fn from(str: &str) -> Self {
        unsafe { mlirStringRefCreateFromCString(str.as_ptr() as *const i8) }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self { ptr: std::ptr::null() }
    }
}
