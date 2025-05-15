use std::io;

use crate::header::DType;
use crate::type_str::TypeStr;

#[cfg(test)]
#[macro_use]
mod test_helpers;

pub use traits::{Serialize, Deserialize, AutoSerialize};
pub use traits::{TypeRead, TypeWrite, TypeWriteDyn, TypeReadDyn, DTypeError};
use traits::{helper, ErrorKind};
#[macro_use]
mod traits;

pub use slice::FixedSizeBytes;
mod slice;

mod primitive;

mod array_member;

// helpers
fn invalid_data<T: ToString>(message: T) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn expect_scalar_dtype<T: ?Sized>(dtype: &DType) -> Result<&TypeStr, DTypeError> {
    dtype.as_scalar().ok_or_else(|| {
        let dtype = dtype.descr();
        let rust_type = std::any::type_name::<T>();
        DTypeError(ErrorKind::ExpectedScalar { dtype, rust_type })
    })
}
