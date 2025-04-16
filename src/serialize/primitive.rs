//! Integers, floats, complex.

#[cfg(feature = "half")]
use half::f16;
use std::io;
use std::marker::PhantomData;

#[cfg(feature = "complex")]
use num_complex::Complex;

use crate::header::DType;
use crate::type_str::{TypeStr, Endianness, TypeChar};
use super::{DTypeError, TypeRead, TypeWrite, Serialize, Deserialize, AutoSerialize};
use super::{expect_scalar_dtype, invalid_data};

/// Implementation detail of reading and writing for primitive types.
pub trait PrimitiveReadWrite: Sized {
    #[doc(hidden)]
    fn primitive_read_one<R: io::Read>(reader: R, swap_bytes: bool) -> io::Result<Self>;
    #[doc(hidden)]
    fn primitive_write_one<W: io::Write>(&self, writer: W, swap_bytes: bool) -> io::Result<()>;
}

macro_rules! derive_int_primitive_read_write {
    ($($int:ident)*) => {$(
        impl PrimitiveReadWrite for $int {
            #[inline]
            fn primitive_read_one<R: io::Read>(mut reader: R, swap_bytes: bool) -> io::Result<$int> {
                use std::mem::size_of;

                let mut buf = [0u8; size_of::<$int>()];
                reader.read_exact(&mut buf)?;

                let out = $int::from_ne_bytes(buf);
                match swap_bytes {
                    true => Ok(out.swap_bytes()),
                    false => Ok(out),
                }
            }

            #[inline]
            fn primitive_write_one<W: io::Write>(&self, mut writer: W, swap_bytes: bool) -> io::Result<()> {
                let swapped = match swap_bytes {
                    true => self.swap_bytes(),
                    false => *self,
                };

                writer.write_all(&swapped.to_ne_bytes())?;

                Ok(())
            }
        }
    )*};
}

macro_rules! derive_float_primitive_read_write {
    ($float:ident as $int:ident) => {
        impl PrimitiveReadWrite for $float {
            #[inline]
            fn primitive_read_one<R: io::Read>(reader: R, swap_bytes: bool) -> io::Result<$float> {
                let bits = <$int>::primitive_read_one(reader, swap_bytes)?;
                Ok(<$float>::from_bits(bits))
            }

            #[inline]
            fn primitive_write_one<W: io::Write>(&self, writer: W, swap_bytes: bool) -> io::Result<()> {
                self.to_bits().primitive_write_one(writer, swap_bytes)
            }
        }
    };
}

derive_int_primitive_read_write!{ u8 u16 u32 u64 }
derive_int_primitive_read_write!{ i8 i16 i32 i64 }
#[cfg(feature = "half")]
derive_float_primitive_read_write!{ f16 as u16 }
derive_float_primitive_read_write!{ f32 as u32 }
derive_float_primitive_read_write!{ f64 as u64 }

impl PrimitiveReadWrite for bool {
    fn primitive_read_one<R: io::Read>(mut reader: R, _swap_bytes: bool) -> io::Result<bool> {
        let mut buf = [0; 1];
        reader.read_exact(&mut buf)?;
        if buf[0] >= 2 {
            return Err(invalid_data(format_args!("invalid value for bool: {}", buf[0])))
        }
        Ok(buf[0] != 0)
    }

    fn primitive_write_one<W: io::Write>(&self, mut writer: W, _swap_bytes: bool) -> io::Result<()> {
        writer.write_all(&[*self as u8])
    }
}

/// Implementation of [`TypeRead`] using [`PrimitiveReadWrite`].
#[doc(hidden)]
pub struct PrimitiveReader<T> {
    swap_bytes: bool,
    _marker: PhantomData<T>
}

/// Implementation of [`TypeWrite`] using [`PrimitiveReadWrite`].
#[doc(hidden)]
pub struct PrimitiveWriter<T> {
    swap_bytes: bool,
    _marker: PhantomData<T>
}

impl<T> PrimitiveReader<T> {
    pub(super) fn new(endianness: Endianness) -> Self {
        PrimitiveReader {
            swap_bytes: endianness.requires_swap(Endianness::of_machine()),
            _marker: PhantomData,
        }
    }
}

impl<T> PrimitiveWriter<T> {
    pub(super) fn new(endianness: Endianness) -> Self {
        PrimitiveWriter {
            swap_bytes: endianness.requires_swap(Endianness::of_machine()),
            _marker: PhantomData,
        }
    }
}

impl<T: PrimitiveReadWrite> TypeRead for PrimitiveReader<T> {
    type Value = T;

    #[inline(always)]
    fn read_one<R: io::Read>(&self, reader: R) -> io::Result<Self::Value> {
        T::primitive_read_one(reader, self.swap_bytes)
    }
}

impl<T: PrimitiveReadWrite> TypeWrite for PrimitiveWriter<T> {
    type Value = T;

    #[inline(always)]
    fn write_one<W: io::Write>(&self, writer: W, value: &Self::Value) -> io::Result<()> {
        value.primitive_write_one(writer, self.swap_bytes)
    }
}

#[cfg(feature = "complex")]
#[doc(hidden)]
pub struct ComplexReader<F> { pub(super) float: PrimitiveReader<F> }
#[cfg(feature = "complex")]
#[doc(hidden)]
pub struct ComplexWriter<F> { pub(super) float: PrimitiveWriter<F> }

#[cfg(feature = "complex")]
impl<F: PrimitiveReadWrite> TypeRead for ComplexReader<F> {
    type Value = Complex<F>;

    #[inline]
    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Self::Value> {
        let re = self.float.read_one(&mut reader)?;
        let im = self.float.read_one(&mut reader)?;
        Ok(Complex { re, im })
    }
}

#[cfg(feature = "complex")]
impl<F: PrimitiveReadWrite> TypeWrite for ComplexWriter<F> {
    type Value = Complex<F>;

    #[inline]
    fn write_one<W: io::Write>(&self, mut writer: W, value: &Complex<F>) -> io::Result<()> {
        self.float.write_one(&mut writer, &value.re)?;
        self.float.write_one(&mut writer, &value.im)?;
        Ok(())
    }
}

macro_rules! impl_primitive_serializable {
    (
        rust: [ $([$size:tt $prim:ty])* ]
        npy: [ (main_ty: $MainTy:path) (support_ty: $SupportTy:pat) ]
    ) => {$(
        impl Deserialize for $prim {
            type TypeReader = PrimitiveReader<$prim>;

            fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
                match expect_scalar_dtype::<Self>(dtype)? {
                    // Read an integer of the correct size and signedness.
                    //
                    // DateTime is an unsigned integer and TimeDelta is a signed integer,
                    // so we support those too.
                    &TypeStr { size: $size, endianness, type_char: $SupportTy, .. } => {
                        Ok(PrimitiveReader::new(endianness))
                    },
                    type_str => Err(DTypeError::bad_scalar::<Self>("read", type_str)),
                }
            }
        }

        impl Serialize for $prim {
            type TypeWriter = PrimitiveWriter<$prim>;

            fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
                match expect_scalar_dtype::<Self>(dtype)? {
                    // Write an integer of the correct size and signedness.
                    &TypeStr { size: $size, endianness, type_char: $SupportTy, .. } => {
                        Ok(PrimitiveWriter::new(endianness))
                    },
                    type_str => Err(DTypeError::bad_scalar::<Self>("write", type_str)),
                }
            }
        }

        impl AutoSerialize for $prim {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness($MainTy, $size, None))
            }
        }
    )*};
}

impl_primitive_serializable! {
    rust: [ [1 i8] [2 i16] [4 i32] ]
    npy: [ (main_ty: TypeChar::Int) (support_ty: TypeChar::Int) ]
}

impl_primitive_serializable! {
    rust: [ [8 i64] ]
    npy: [ (main_ty: TypeChar::Int) (support_ty: TypeChar::Int | TypeChar::TimeDelta | TypeChar::DateTime) ]
}

impl_primitive_serializable! {
    rust: [ [1 u8] [2 u16] [4 u32] [8 u64] ]
    npy: [ (main_ty: TypeChar::Uint) (support_ty: TypeChar::Uint) ]
}

// TODO: numpy supports f128
impl_primitive_serializable! {
    rust: [ [4 f32] [8 f64] ]
    npy: [ (main_ty: TypeChar::Float) (support_ty: TypeChar::Float) ]
}
#[cfg(feature = "half")]
impl_primitive_serializable! {
    rust: [ [2 f16] ]
    npy: [ (main_ty: TypeChar::Float) (support_ty: TypeChar::Float) ]
}

impl_primitive_serializable! {
    rust: [ [1 bool] ]
    npy: [ (main_ty: TypeChar::Bool) (support_ty: TypeChar::Bool) ]
}

macro_rules! impl_complex_serializable {
    ( $( [ $size:literal $float:ident ] )+ ) => { $(
        #[cfg(feature = "complex")]
        /// _This impl is only available with the **`"complex"`** feature._
        impl Deserialize for Complex<$float> {
            type TypeReader = ComplexReader<$float>;

            fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
                const SIZE: u64 = 2 * $size;

                match expect_scalar_dtype::<Self>(dtype)? {
                    &TypeStr { size: SIZE, endianness, type_char: TypeChar::Complex, .. } => {
                        Ok(ComplexReader { float: PrimitiveReader::new(endianness) })
                    },
                    type_str => Err(DTypeError::bad_scalar::<Self>("read", type_str)),
                }
            }
        }

        #[cfg(feature = "complex")]
        /// _This impl is only available with the **`"complex"`** feature._
        impl Serialize for Complex<$float> {
            type TypeWriter = ComplexWriter<$float>;

            fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
                const SIZE: u64 = 2 * $size;

                match expect_scalar_dtype::<Self>(dtype)? {
                    &TypeStr { size: SIZE, endianness, type_char: TypeChar::Complex, .. } => {
                        Ok(ComplexWriter { float: PrimitiveWriter::new(endianness) })
                    },
                    type_str => Err(DTypeError::bad_scalar::<Self>("write", type_str)),
                }
            }
        }

        #[cfg(feature = "complex")]
        /// _This impl is only available with the **`"complex"`** feature._
        impl AutoSerialize for Complex<$float> {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness(TypeChar::Complex, 2 * $size, None))
            }
        }
    )+};
}

impl_complex_serializable! { [ 4 f32 ] [ 8 f64 ] }


#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;
    use crate::serialize::test_helpers::*;

    #[test]
    fn identity() {
        let be = DType::parse("'>i4'").unwrap();
        let le = DType::parse("'<i4'").unwrap();

        assert_eq!(reader_output::<i32>(&be, &blob![be(1_i32)]), 1);
        assert_eq!(reader_output::<i32>(&le, &blob![le(1_i32)]), 1);
        assert_eq!(writer_output::<i32>(&be, &1), blob![be(1_i32)]);
        assert_eq!(writer_output::<i32>(&le, &1), blob![le(1_i32)]);

        let be = DType::parse("'>u4'").unwrap();
        let le = DType::parse("'<u4'").unwrap();

        assert_eq!(reader_output::<u32>(&be, &blob![be(1_u32)]), 1);
        assert_eq!(reader_output::<u32>(&le, &blob![le(1_u32)]), 1);
        assert_eq!(writer_output::<u32>(&be, &1), blob![be(1_u32)]);
        assert_eq!(writer_output::<u32>(&le, &1), blob![le(1_u32)]);

        for &dtype in &["'>i1'", "'<i1'", "'|i1'"] {
            let dtype = DType::parse(dtype).unwrap();
            assert_eq!(reader_output::<i8>(&dtype, &blob![1]), 1);
            assert_eq!(writer_output::<i8>(&dtype, &1), blob![1]);
        }

        for &dtype in &["'>u1'", "'<u1'", "'|u1'"] {
            let dtype = DType::parse(dtype).unwrap();
            assert_eq!(reader_output::<u8>(&dtype, &blob![1]), 1);
            assert_eq!(writer_output::<u8>(&dtype, &1), blob![1]);
        }
    }

    #[test]
    fn native_float_types() {
        let be_bytes = 42.0_f64.to_bits().to_be_bytes();
        let le_bytes = 42.0_f64.to_bits().to_le_bytes();
        let be = DType::parse("'>f8'").unwrap();
        let le = DType::parse("'<f8'").unwrap();

        assert_eq!(reader_output::<f64>(&be, &be_bytes), 42.0);
        assert_eq!(reader_output::<f64>(&le, &le_bytes), 42.0);
        assert_eq!(writer_output::<f64>(&be, &42.0), &be_bytes);
        assert_eq!(writer_output::<f64>(&le, &42.0), &le_bytes);

        let be_bytes = 42.0_f32.to_bits().to_be_bytes();
        let le_bytes = 42.0_f32.to_bits().to_le_bytes();
        let be = DType::parse("'>f4'").unwrap();
        let le = DType::parse("'<f4'").unwrap();

        assert_eq!(reader_output::<f32>(&be, &be_bytes), 42.0);
        assert_eq!(reader_output::<f32>(&le, &le_bytes), 42.0);
        assert_eq!(writer_output::<f32>(&be, &42.0), &be_bytes);
        assert_eq!(writer_output::<f32>(&le, &42.0), &le_bytes);
    }

    #[test]
    fn native_bool() {
        assert!(DType::parse("'|b2'").is_err());
        let dtype = DType::parse("'|b1'").unwrap();

        assert_eq!(reader_output::<bool>(&dtype, &[0]), false);
        assert_eq!(reader_output::<bool>(&dtype, &[1]), true);
        reader_expect_read_err::<bool>(&dtype, &[2]);
        reader_expect_read_err::<bool>(&dtype, &[255]);

        assert_eq!(writer_output::<bool>(&dtype, &false), &[0]);
        assert_eq!(writer_output::<bool>(&dtype, &true), &[1]);
    }

    #[test]
    #[cfg(feature = "half")]
    fn native_half_types() {
        use half::f16;

        let c = f16::from_f32_const(42.69);
        let be_bytes = blob![be(c.to_bits())];
        let le_bytes = blob![le(c.to_bits())];

        let be = DType::parse(&format!("'>f2'")).unwrap();
        let le = DType::parse(&format!("'<f2'")).unwrap();

        assert_eq!(reader_output::<f16>(&be, &be_bytes), c);
        assert_eq!(reader_output::<f16>(&le, &le_bytes), c);
        assert_eq!(writer_output::<f16>(&be, &c), be_bytes);
        assert_eq!(writer_output::<f16>(&le, &c), le_bytes);

        let c = f16::from_f32_const(42.69);
        let be_bytes = blob![be(c.to_bits())];
        let le_bytes = blob![le(c.to_bits())];

        let be = DType::parse(&format!("'>f2'")).unwrap();
        let le = DType::parse(&format!("'<f2'")).unwrap();

        assert_eq!(reader_output::<f16>(&be, &be_bytes), c);
        assert_eq!(reader_output::<f16>(&le, &le_bytes), c);
        assert_eq!(writer_output::<f16>(&be, &c), be_bytes);
        assert_eq!(writer_output::<f16>(&le, &c), le_bytes);
    }

    #[test]
    #[cfg(feature = "complex")]
    fn native_complex_types() {
        use num_complex::{Complex32, Complex64};

        let c = Complex64 { re: 42.0, im: 63.0 };
        let be_bytes = blob![be(c.re.to_bits()), be(c.im.to_bits())];
        let le_bytes = blob![le(c.re.to_bits()), le(c.im.to_bits())];

        let be = DType::parse(&format!("'>c16'")).unwrap();
        let le = DType::parse(&format!("'<c16'")).unwrap();

        assert_eq!(reader_output::<Complex64>(&be, &be_bytes), c);
        assert_eq!(reader_output::<Complex64>(&le, &le_bytes), c);
        assert_eq!(writer_output::<Complex64>(&be, &c), be_bytes);
        assert_eq!(writer_output::<Complex64>(&le, &c), le_bytes);

        let c = Complex32 { re: 42.0, im: 63.0 };
        let be_bytes = blob![be(c.re.to_bits()), be(c.im.to_bits())];
        let le_bytes = blob![le(c.re.to_bits()), le(c.im.to_bits())];

        let be = DType::parse(&format!("'>c8'")).unwrap();
        let le = DType::parse(&format!("'<c8'")).unwrap();

        assert_eq!(reader_output::<Complex32>(&be, &be_bytes), c);
        assert_eq!(reader_output::<Complex32>(&le, &le_bytes), c);
        assert_eq!(writer_output::<Complex32>(&be, &c), be_bytes);
        assert_eq!(writer_output::<Complex32>(&le, &c), le_bytes);
    }

    #[test]
    fn datetime_as_int() {
        let be = DType::parse("'>m8[ns]'").unwrap();
        let le = DType::parse("'<m8[ns]'").unwrap();

        assert_eq!(reader_output::<i64>(&be, &blob![be(1_i64)]), 1);
        assert_eq!(reader_output::<i64>(&le, &blob![le(1_i64)]), 1);
        assert_eq!(writer_output::<i64>(&be, &1), blob![be(1_i64)]);
        assert_eq!(writer_output::<i64>(&le, &1), blob![le(1_i64)]);

        let be = DType::parse("'>M8[ns]'").unwrap();
        let le = DType::parse("'<M8[ns]'").unwrap();

        assert_eq!(reader_output::<i64>(&be, &blob![be(1_i64)]), 1);
        assert_eq!(reader_output::<i64>(&le, &blob![le(1_i64)]), 1);
        assert_eq!(writer_output::<i64>(&be, &1), blob![be(1_i64)]);
        assert_eq!(writer_output::<i64>(&le, &1), blob![le(1_i64)]);
    }

    #[test]
    fn bad_datetime_types() {
        // the "size must be 8" restriction is part of DType parsing
        assert!(DType::parse("'>m8[ns]'").is_ok());
        assert!(DType::parse("'>m4[ns]'").is_err());
        assert!(DType::parse("'>M4[ns]'").is_err());

        // must be signed
        let datetime = DType::parse("'<M8[ns]'").unwrap();
        let timedelta = DType::parse("'<m8[ns]'").unwrap();
        reader_expect_ok::<i64>(&datetime);
        reader_expect_err::<u64>(&datetime);
        reader_expect_ok::<i64>(&timedelta);
        reader_expect_err::<u64>(&timedelta);
    }

    #[test]
    fn wrong_size_int() {
        let t_i32 = DType::parse("'<i4'").unwrap();
        let t_u32 = DType::parse("'<u4'").unwrap();

        reader_expect_err::<i64>(&t_i32);
        reader_expect_err::<i16>(&t_i32);
        reader_expect_err::<u64>(&t_u32);
        reader_expect_err::<u16>(&t_u32);
        writer_expect_err::<i64>(&t_i32);
        writer_expect_err::<i16>(&t_i32);
        writer_expect_err::<u64>(&t_u32);
        writer_expect_err::<u16>(&t_u32);
    }

    #[test]
    fn default_simple_type_strs() {
        assert_eq!(i8::default_dtype().descr(), "'|i1'");
        assert_eq!(u8::default_dtype().descr(), "'|u1'");

        if 1 == i32::from_be(1) {
            assert_eq!(i16::default_dtype().descr(), "'>i2'");
            assert_eq!(i32::default_dtype().descr(), "'>i4'");
            assert_eq!(i64::default_dtype().descr(), "'>i8'");
            assert_eq!(u32::default_dtype().descr(), "'>u4'");
        } else {
            assert_eq!(i16::default_dtype().descr(), "'<i2'");
            assert_eq!(i32::default_dtype().descr(), "'<i4'");
            assert_eq!(i64::default_dtype().descr(), "'<i8'");
            assert_eq!(u32::default_dtype().descr(), "'<u4'");
        }
    }

    #[test]
    #[cfg(feature = "complex")]
    fn default_complex_type_strs() {
        assert_eq!(Complex::<f32>::default_dtype().descr(), "'<c8'");
        assert_eq!(Complex::<f64>::default_dtype().descr(), "'<c16'");
    }
}
