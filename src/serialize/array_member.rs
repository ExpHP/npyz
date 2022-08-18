use std::io;

use crate::header::DType;
use super::{DTypeError, ErrorKind, TypeRead, TypeWrite, Serialize, Deserialize, AutoSerialize};

impl DType {
    /// Expect an array dtype, get the inner dtype.
    fn array_inner_dtype(&self, expected_len: u64) -> Result<&Self, DTypeError> {
        match *self {
            DType::Record { .. } => Err(DTypeError(ErrorKind::ExpectedArray { got: "a record" })),
            DType::Plain { .. } => Err(DTypeError(ErrorKind::ExpectedArray { got: "a scalar" })),
            DType::Array(len, ref ty) => {
                if len != expected_len {
                    return Err(DTypeError(ErrorKind::WrongArrayLen {
                        actual: len,
                        expected: expected_len,
                    }));
                }

                Ok(ty)
            },
        }
    }
}

pub struct ArrayReader<I, const N: usize>{ inner: I }
pub struct ArrayWriter<I, const N: usize>{ inner: I }

impl<I: TypeRead, const N: usize> TypeRead for ArrayReader<I, N>
where I::Value: Copy + Default,
{
    type Value = [I::Value; N];

    #[inline]
    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Self::Value> {
        let mut value = [I::Value::default(); N];
        for place in &mut value {
            *place = self.inner.read_one(&mut reader)?;
        }

        Ok(value)
    }
}

impl<I: TypeWrite, const N: usize> TypeWrite for ArrayWriter<I, N>
where I::Value: Sized,
{
    type Value = [I::Value; N];

    #[inline]
    fn write_one<W: io::Write>(&self, mut writer: W, value: &Self::Value) -> io::Result<()>
    where Self: Sized,
    {
        for item in value {
            self.inner.write_one(&mut writer, item)?;
        }
        Ok(())
    }
}

impl<T: AutoSerialize + Default + Copy, const N: usize> AutoSerialize for [T; N] {
    #[inline]
    fn default_dtype() -> DType {
        DType::Array(N as u64, Box::new(T::default_dtype()))
    }
}

impl<T: Deserialize + Default + Copy, const N: usize> Deserialize for [T; N] {
    type TypeReader = ArrayReader<<T as Deserialize>::TypeReader, N>;

    #[inline]
    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let inner_dtype = dtype.array_inner_dtype(N as u64)?;
        let inner = <T>::reader(inner_dtype)?;
        Ok(ArrayReader { inner })
    }
}

impl<T: Serialize, const N: usize> Serialize for [T; N] {
    type TypeWriter = ArrayWriter<<T as Serialize>::TypeWriter, N>;

    #[inline]
    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let inner = <T>::writer(dtype.array_inner_dtype(N as u64)?)?;
        Ok(ArrayWriter { inner })
    }
}

// NOTE: Tests for arrays are in tests/serialize_array.rs because they require derives
