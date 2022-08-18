use crate::header::DType;
use super::{TypeRead, TypeWrite, Serialize, Deserialize};

/// Helper to generate a blob of bytes.
macro_rules! blob {
    // int literals are taken to be u8
    (@state [$($encoded_parts:tt)*] $byte:literal $(, $($rest:tt)* )?) => {
        blob!{@state [ $($encoded_parts)* [[$byte]] ] $($($rest)*)?}
    };
    // le(EXPR) encodes as little endian
    (@state [$($encoded_parts:tt)*] le($val:expr) $(, $($rest:tt)* )?) => {
        blob!{@state [ $($encoded_parts)* [$val.to_le_bytes()] ] $($($rest)*)?}
    };
    // be(EXPR) encodes as big endian
    (@state [$($encoded_parts:tt)*] be($val:expr) $(, $($rest:tt)* )?) => {
        blob!{@state [ $($encoded_parts)* [$val.to_be_bytes()] ] $($($rest)*)?}
    };
    // finish
    (@state [ $([ $($encoded_parts:tt)+ ])* ] ) => {{
        #[allow(unused_mut)]
        let mut blob: Vec<u8> = vec![];
        $( blob.extend_from_slice(& $($encoded_parts)+ [..]); )*
        blob
    }};
    // begin
    ($($input:tt)*) => { blob!{ @state [] $($input)* } };
}

pub use helper_funcs::*;

mod helper_funcs {
    // some funcs might only be used by a feature flag's tests and it can be annoying to pin down
    #![allow(unused)]

    use super::*;

    pub fn reader_output<T: Deserialize + core::fmt::Debug>(dtype: &DType, bytes: &[u8]) -> T {
        let type_reader = T::reader(dtype).unwrap_or_else(|e| panic!("{}", e));

        let mut reader = bytes;
        let value = type_reader.read_one(&mut reader).expect("reader_output failed");
        assert_eq!(reader.len(), 0, "reader did not read all bytes");
        value
    }

    pub fn reader_expect_ok<T: Deserialize>(dtype: &DType) {
        assert!(T::reader(dtype).is_ok())
    }
    pub fn reader_expect_err<T: Deserialize>(dtype: &DType) {
        T::reader(dtype).err().expect("reader_expect_err failed!");
    }
    pub fn reader_expect_read_ok<T: Deserialize>(dtype: &DType, bytes: &[u8]) {
        let mut reader = bytes;
        T::reader(dtype).unwrap_or_else(|e| panic!("{}", e))
            .read_one(&mut reader)
            .ok().expect("reader_expect_read_ok failed!");
        assert_eq!(reader.len(), 0, "reader did not read all bytes");
    }
    pub fn reader_expect_read_err<T: Deserialize>(dtype: &DType, bytes: &[u8]) {
        T::reader(dtype).unwrap_or_else(|e| panic!("{}", e))
            .read_one(bytes)
            .err().expect("reader_expect_read_err failed!");
    }

    pub fn writer_output<T: Serialize + ?Sized>(dtype: &DType, value: &T) -> Vec<u8> {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value).unwrap();
        vec
    }

    pub fn writer_expect_ok<T: Serialize + ?Sized>(dtype: &DType) {
        assert!(T::writer(dtype).is_ok())
    }
    pub fn writer_expect_err<T: Serialize + ?Sized>(dtype: &DType) {
        T::writer(dtype).err().expect("writer_expect_err failed!");
    }
    pub fn writer_expect_write_ok<T: Serialize + ?Sized>(dtype: &DType, value: &T) {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value)
            .ok().expect("writer_expect_write_ok failed!");
    }
    pub fn writer_expect_write_err<T: Serialize + ?Sized>(dtype: &DType, value: &T) {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value)
            .err().expect("writer_expect_write_err failed!");
    }
}

