use crate::codegen::Codegen;
use crate::field::P;
use llzk_bridge::{
    AssemblyFormatData, BytecodeFormatData, FormatData, OutputFormat_OF_Assembly,
    OutputFormat_OF_Bytecode, OutputFormat_OF_Picus, PicusFormatData,
};
use std::fmt;
use std::ptr;
use std::slice;

#[repr(u32)]
#[derive(PartialEq, Clone, Debug)]
pub enum OutputFormats {
    Assembly = OutputFormat_OF_Assembly,
    Bytecode = OutputFormat_OF_Bytecode,
    Picus = OutputFormat_OF_Picus,
}

impl Default for OutputFormats {
    fn default() -> Self {
        Self::Bytecode
    }
}

impl Into<FormatData> for OutputFormats {
    fn into(self) -> FormatData {
        match self {
            OutputFormats::Assembly => AssemblyFormatData::default().into(),
            OutputFormats::Bytecode => BytecodeFormatData::default().into(),
            OutputFormats::Picus => PicusFormatData { prime: P }.into(),
        }
    }
}

/// Final output type of the llzk code generator.
pub struct CodegenOutput {
    bytes: *mut u8,
    size: usize,
    format: OutputFormats,
}

impl CodegenOutput {
    pub fn new(format: OutputFormats) -> Self {
        Self { bytes: ptr::null_mut(), size: 0, format }
    }

    pub fn bytes_ref_mut(&mut self) -> &mut *mut u8 {
        &mut self.bytes
    }

    pub fn size_ref_mut(&mut self) -> &mut usize {
        &mut self.size
    }

    pub fn bytes(&self) -> *mut u8 {
        self.bytes
    }
}

impl Default for CodegenOutput {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl Drop for CodegenOutput {
    fn drop(&mut self) {
        let codegen = Codegen::instance().unwrap();
        codegen.release_output(self);
    }
}

impl AsRef<[u8]> for CodegenOutput {
    fn as_ref(&self) -> &[u8] {
        if self.bytes.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.bytes, self.size) }
    }
}

fn do_row<F: FnMut(&u8, &mut fmt::Formatter) -> fmt::Result>(
    row: &[u8],
    mut cb: F,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    for part in row.chunks(8) {
        for byte in part {
            cb(byte, f)?;
        }
        write!(f, " ")?;
    }
    Ok(())
}

fn hexdump<B>(data: B, f: &mut fmt::Formatter) -> fmt::Result
where
    B: AsRef<[u8]>,
{
    let mut cursor: u32 = 0;
    for row in data.as_ref().chunks(16) {
        write!(f, "{cursor:#10x}: ")?;
        do_row(row, |byte, f| write!(f, "{byte:x} "), f)?;
        write!(f, "|")?;
        do_row(
            row,
            |byte, f| {
                if byte.is_ascii() {
                    write!(f, "{}", *byte as char)
                } else {
                    write!(f, ".")
                }
            },
            f,
        )?;
        write!(f, "|")?;
        cursor += 16;
    }
    Ok(())
}

impl fmt::Display for CodegenOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.format == OutputFormats::Bytecode {
            writeln!(f, "Bytecode hexdump:")?;
            let data: &[u8] = self.as_ref();
            return hexdump(data, f);
        }
        let str = std::str::from_utf8(unsafe { std::slice::from_raw_parts(self.bytes, self.size) })
            .unwrap();
        write!(f, "{str}")
    }
}
