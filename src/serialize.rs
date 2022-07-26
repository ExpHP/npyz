use std::io;
use std::fmt;
use std::convert::TryFrom;
use std::marker::PhantomData;

#[cfg(feature = "complex")]
use num_complex::Complex;

use crate::header::DType;
use crate::type_str::{TypeStr, Endianness, TypeKind};

#[cfg(feature = "arraystring")]
use arrayvec::ArrayString;


/// Trait that permits reading a type from an `.npy` file.
///
/// Examples of types that implement this:
///
/// * Primitive integers, floats, `Complex` (with the **`"complex"`** feature)
/// * Owned byte containers (`Vec<u8>`)
///
/// _This trait is derivable when enabling the **`"derive"`** feature._ This makes it easier
/// to work with structured arrays.
///
/// For an example of how to implement this manually, see `Vector5` in the
/// [roundtrip test](https://github.com/ExpHP/npyz/tree/master/tests/roundtrip.rs).
pub trait Deserialize: Sized {
    /// Think of this as like a `for<R: io::Read> Fn(R) -> io::Result<Self>`.
    ///
    /// There is no closure-like sugar for these; you must manually define a type that
    /// implements [`TypeRead`].
    type TypeReader: TypeRead<Value=Self>;

    /// Get a function that deserializes a single data field at a time.
    ///
    /// The purpose of the `dtype` arugment is to allow e.g. specifying a length for string types,
    /// or the endianness for integers.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `DType` is not compatible with `Self`.
    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError>;
}

/// Trait that permits writing a type to an `.npy` file.
///
/// Examples of types that implement this:
///
/// * Primitive integers, floats, `Complex` (with the **`"complex"`** feature)
/// * Byte slices (`[u8]`)
///
/// _This trait is derivable when enabling the **`"derive"`** feature._ This makes it easier
/// to work with structured arrays.
///
/// For an example of how to implement this manually, see `Vector5` in the
/// [roundtrip test](https://github.com/ExpHP/npyz/tree/master/tests/roundtrip.rs).
pub trait Serialize {
    /// Think of this as some sort of `for<W: io::Write> Fn(W, &Self) -> io::Result<()>`.
    ///
    /// There is no closure-like sugar for these; you must manually define a type that
    /// implements [`TypeWrite`].
    type TypeWriter: TypeWrite<Value=Self>;

    /// Get a function that serializes a single data field at a time.
    ///
    /// The purpose of the `dtype` arugment is to allow e.g. specifying a length for string types,
    /// or the endianness for integers.  The derivable [`AutoSerialize`] trait is able to supply
    /// many types with a reasonable default.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `DType` is not compatible with `Self`.
    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError>;
}

/// Subtrait of [`Serialize`] for types which have a reasonable default [`DType`].
///
/// This opens up some simpler APIs for serialization. (e.g. [`crate::to_file`], [`crate::WriterBuilder::default_dtype`])
///
/// _This trait is derivable when enabling the **`"derive"`** feature._ This makes it easier
/// to work with structured arrays.
///
/// For an example of how to implement this manually, see `Vector5` in the
/// [roundtrip test](https://github.com/ExpHP/npyz/tree/master/tests/roundtrip.rs).
pub trait AutoSerialize: Serialize {
    /// A suggested format for serialization.
    ///
    /// The builtin implementations for primitive types generally prefer `|` endianness if possible,
    /// else the machine endian format.
    fn default_dtype() -> DType;
}

/// Like some sort of `for<R: io::Read> Fn(R) -> io::Result<T>`.
///
/// For an example of how to implement this manually, see `Vector5` in the
/// [roundtrip test](https://github.com/ExpHP/npyz/tree/master/tests/roundtrip.rs).
///
/// # Trait objects
///
/// `dyn TypeRead` has no object-safe methods.
/// If you need dynamic polymorphism, use `dyn` [`TypeReadDyn`] instead.
pub trait TypeRead {
    /// Type returned by the function.
    type Value;

    /// The function.
    fn read_one<R: io::Read>(&self, bytes: R) -> io::Result<Self::Value> where Self: Sized;
}

/// Like some sort of `for<W: io::Write> Fn(W, &T) -> io::Result<()>`.
///
/// For an example of how to implement this manually, see `Vector5` in the
/// [roundtrip test](https://github.com/ExpHP/npyz/tree/master/tests/roundtrip.rs).
///
/// # Trait objects
///
/// `dyn TypeWrite` has no object-safe methods.
/// If you need dynamic polymorphism, use `dyn` [`TypeWriteDyn`] instead.
pub trait TypeWrite {
    /// Type accepted by the function.
    type Value: ?Sized;

    /// The function.
    fn write_one<W: io::Write>(&self, writer: W, value: &Self::Value) -> io::Result<()>
    where Self: Sized;
}

/// The proper trait to use for trait objects of [`TypeRead`].
///
/// `Box<dyn TypeRead>` is useless because `dyn TypeRead` has no object-safe methods.
/// The workaround is to use `Box<dyn TypeReadDyn>` instead, which itself implements `TypeRead`.
pub trait TypeReadDyn: TypeRead {
    #[doc(hidden)]
    fn read_one_dyn(&self, writer: &mut dyn io::Read) -> io::Result<Self::Value>;
}

impl<T: TypeRead> TypeReadDyn for T {
    #[inline(always)]
    fn read_one_dyn(&self, reader: &mut dyn io::Read) -> io::Result<Self::Value> {
        self.read_one(reader)
    }
}

/// The proper trait to use for trait objects of [`TypeWrite`].
///
/// `Box<dyn TypeWrite>` is useless because `dyn TypeWrite` has no object-safe methods.
/// The workaround is to use `Box<dyn TypeWriteDyn>` instead, which itself implements `TypeWrite`.
pub trait TypeWriteDyn: TypeWrite {
    #[doc(hidden)]
    fn write_one_dyn(&self, writer: &mut dyn io::Write, value: &Self::Value) -> io::Result<()>;
}

impl<T: TypeWrite> TypeWriteDyn for T {
    #[inline(always)]
    fn write_one_dyn(&self, writer: &mut dyn io::Write, value: &Self::Value) -> io::Result<()> {
        self.write_one(writer, value)
    }
}

/// Indicates that a particular rust type does not support serialization or deserialization
/// as a given [`DType`].
#[derive(Debug, Clone)]
pub struct DTypeError(ErrorKind);

#[derive(Debug, Clone)]
enum ErrorKind {
    Custom(String),
    ExpectedScalar {
        dtype: String,
        rust_type: &'static str,
    },
    ExpectedArray {
        got: &'static str, // "a scalar", "a record"
    },
    WrongArrayLen {
        expected: u64,
        actual: u64,
    },
    ExpectedRecord {
        dtype: String,
    },
    WrongFields {
        expected: Vec<String>,
        actual: Vec<String>,
    },
    BadScalar {
        type_str: TypeStr,
        rust_type: &'static str,
        verb: &'static str,
    },
    UsizeOverflow(u64),
}

impl std::error::Error for DTypeError {}

impl DTypeError {
    /// Construct with a custom error message.
    pub fn custom<S: AsRef<str>>(msg: S) -> Self {
        DTypeError(ErrorKind::Custom(msg.as_ref().to_string()))
    }

    // verb should be "read" or "write"
    fn bad_scalar(verb: &'static str, type_str: &TypeStr, rust_type: &'static str) -> Self {
        let type_str = type_str.clone();
        DTypeError(ErrorKind::BadScalar { type_str, rust_type, verb })
    }

    fn expected_scalar(dtype: &DType, rust_type: &'static str) -> Self {
        let dtype = dtype.descr();
        DTypeError(ErrorKind::ExpectedScalar { dtype, rust_type })
    }

    fn bad_usize(x: u64) -> Self {
        DTypeError(ErrorKind::UsizeOverflow(x))
    }

    // used by derives
    #[doc(hidden)]
    pub fn expected_record(dtype: &DType) -> Self {
        DTypeError(ErrorKind::ExpectedRecord { dtype: dtype.descr() })
    }

    // used by derives
    #[doc(hidden)]
    pub fn wrong_fields<S1: AsRef<str>, S2: AsRef<str>>(
        expected: impl IntoIterator<Item=S1>,
        actual: impl IntoIterator<Item=S2>,
    ) -> Self {
        DTypeError(ErrorKind::WrongFields {
            expected: expected.into_iter().map(|s| s.as_ref().to_string()).collect(),
            actual: actual.into_iter().map(|s| s.as_ref().to_string()).collect(),
        })
    }
}

impl fmt::Display for DTypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            ErrorKind::Custom(msg) => {
                write!(f, "{}", msg)
            },
            ErrorKind::ExpectedScalar { dtype, rust_type } => {
                write!(f, "type {} requires a scalar (string) dtype, not {}", rust_type, dtype)
            },
            ErrorKind::ExpectedRecord { dtype } => {
                write!(f, "expected a record type; got {}", dtype)
            },
            ErrorKind::ExpectedArray { got } => {
                write!(f, "rust array types require an array dtype (got {})", got)
            },
            ErrorKind::WrongArrayLen { actual, expected } => {
                write!(f, "wrong array size (expected {}, got {})", expected, actual)
            },
            ErrorKind::WrongFields { actual, expected } => {
                write!(f, "field names do not match (expected {:?}, got {:?})", expected, actual)
            },
            ErrorKind::BadScalar { type_str, rust_type, verb } => {
                write!(f, "cannot {} type {} with type-string '{}'", verb, rust_type, type_str)
            },
            ErrorKind::UsizeOverflow(value) => {
                write!(f, "cannot cast {} as usize", value)
            },
        }
    }
}

impl<T> TypeRead for Box<dyn TypeReadDyn<Value=T>> {
    type Value = T;

    #[inline(always)]
    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<T> where Self: Sized {
        (**self).read_one_dyn(&mut reader)
    }
}

impl<T: ?Sized> TypeWrite for Box<dyn TypeWriteDyn<Value=T>> {
    type Value = T;

    #[inline(always)]
    fn write_one<W: io::Write>(&self, mut writer: W, value: &T) -> io::Result<()>
    where Self: Sized,
    {
        // Boxes must always go through two virtual dispatches.
        //
        // (one on the TypeWrite trait object, and one on the Writer which must be
        //  cast to the monomorphic type `&mut dyn io::write`)
        (**self).write_one_dyn(&mut writer, value)
    }
}

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
derive_float_primitive_read_write!{ f32 as u32 }
derive_float_primitive_read_write!{ f64 as u64 }

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
    pub(super) fn new(swap_bytes: bool) -> Self {
        PrimitiveReader {
            swap_bytes,
            _marker: PhantomData,
        }
    }
}

impl<T> PrimitiveWriter<T> {
    pub(super) fn new(swap_bytes: bool) -> Self {
        PrimitiveWriter {
            swap_bytes,
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
pub struct ComplexReader<F> { pub(super) float: PrimitiveReader<F> }
#[cfg(feature = "complex")]
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

fn invalid_data<T>(message: &str) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message.to_string()))
}

fn expect_scalar_dtype<'a>(dtype: &'a DType, rust_type: &'static str) -> Result<&'a TypeStr, DTypeError> {
    dtype.as_scalar().ok_or_else(|| {
        DTypeError::expected_scalar(dtype, rust_type)
    })
}

macro_rules! impl_integer_serializable {
    (
        meta: [ (main_ty: $Int:path) (date_ty: $DateTime:path) ]
        ints: [ $([$size:tt $int:ty])* ]
    ) => {$(
        impl Deserialize for $int {
            type TypeReader = PrimitiveReader<$int>;

            fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
                match expect_scalar_dtype(dtype, stringify!($int))? {
                    // Read an integer of the correct size and signedness.
                    //
                    // DateTime is an unsigned integer and TimeDelta is a signed integer,
                    // so we support those too.
                    TypeStr { size: $size, endianness, type_kind: $Int, .. } |
                    TypeStr { size: $size, endianness, type_kind: $DateTime, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok(PrimitiveReader::new(swap_byteorder))
                    },
                    type_str => Err(DTypeError::bad_scalar("read", type_str, stringify!($int))),
                }
            }
        }

        impl Serialize for $int {
            type TypeWriter = PrimitiveWriter<$int>;

            fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
                match expect_scalar_dtype(dtype, stringify!($int))? {
                    // Write an integer of the correct size and signedness.
                    TypeStr { size: $size, endianness, type_kind: $Int, .. } |
                    TypeStr { size: $size, endianness, type_kind: $DateTime, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok(PrimitiveWriter::new(swap_byteorder))
                    },
                    type_str => Err(DTypeError::bad_scalar("write", type_str, stringify!($int))),
                }
            }
        }

        impl AutoSerialize for $int {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness($Int, $size, None))
            }
        }
    )*};
}

impl_integer_serializable! {
    meta: [ (main_ty: TypeKind::Int) (date_ty: TypeKind::TimeDelta) ]
    ints: [ [1 i8] [2 i16] [4 i32] [8 i64] ]
}

impl_integer_serializable! {
    meta: [ (main_ty: TypeKind::Uint) (date_ty: TypeKind::DateTime) ]
    ints: [ [1 u8] [2 u16] [4 u32] [8 u64] ]
}

// Takes info about each data size, from largest to smallest.
macro_rules! impl_float_serializable {
    ( $( [ $size:literal $float:ident ] )+ ) => { $(
        impl Deserialize for $float {
            type TypeReader = PrimitiveReader<$float>;

            fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
                match expect_scalar_dtype(dtype, stringify!($float))? {
                    // Read a float of the correct size
                    TypeStr { size: $size, endianness, type_kind: TypeKind::Float, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok(PrimitiveReader::new(swap_byteorder))
                    },
                    type_str => Err(DTypeError::bad_scalar("read", type_str, stringify!($float))),
                }
            }
        }

        impl Serialize for $float {
            type TypeWriter = PrimitiveWriter<$float>;

            fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
                match expect_scalar_dtype(dtype, stringify!($float))? {
                    // Write a float of the correct size
                    TypeStr { size: $size, endianness, type_kind: TypeKind::Float, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok(PrimitiveWriter::new(swap_byteorder))
                    },
                    type_str => Err(DTypeError::bad_scalar("write", type_str, stringify!($float))),
                }
            }
        }

        impl AutoSerialize for $float {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness(TypeKind::Float, $size, None))
            }
        }

        #[cfg(feature = "complex")]
        /// _This impl is only available with the **`complex`** feature._
        impl Deserialize for Complex<$float> {
            type TypeReader = ComplexReader<$float>;

            fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
                const SIZE: u64 = 2 * $size;

                match expect_scalar_dtype(dtype, stringify!(Complex<$float>))? {
                    TypeStr { size: SIZE, endianness, type_kind: TypeKind::Complex, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok(ComplexReader { float: PrimitiveReader::new(swap_byteorder) })
                    },
                    type_str => Err(DTypeError::bad_scalar("read", type_str, stringify!(Complex<$float>))),
                }
            }
        }

        #[cfg(feature = "complex")]
        /// _This impl is only available with the **`complex`** feature._
        impl Serialize for Complex<$float> {
            type TypeWriter = ComplexWriter<$float>;

            fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
                const SIZE: u64 = 2 * $size;

                match expect_scalar_dtype(dtype, stringify!(Complex<$float>))? {
                    TypeStr { size: SIZE, endianness, type_kind: TypeKind::Complex, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok(ComplexWriter { float: PrimitiveWriter::new(swap_byteorder) })
                    },
                    type_str => Err(DTypeError::bad_scalar("write", type_str, stringify!(Complex<$float>))),
                }
            }
        }

        #[cfg(feature = "complex")]
        /// _This impl is only available with the **`complex`** feature._
        impl AutoSerialize for Complex<$float> {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness(TypeKind::Complex, $size, None))
            }
        }
    )+};
}

// TODO: numpy supports f16, f128
impl_float_serializable! { [ 4 f32 ] [ 8 f64 ] }

pub struct BytesReader {
    size: usize,
    is_byte_str: bool,
}

impl TypeRead for BytesReader {
    type Value = Vec<u8>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Vec<u8>> {
        let mut vec = vec![0; self.size];
        reader.read_exact(&mut vec)?;

        // truncate trailing zeros for type 'S'
        if self.is_byte_str {
            let end = vec.iter().rposition(|x| x != &0).map_or(0, |ind| ind + 1);
            vec.truncate(end);
        }

        Ok(vec)
    }
}

impl Deserialize for Vec<u8> {
    type TypeReader = BytesReader;

    fn reader(type_str: &DType) -> Result<Self::TypeReader, DTypeError> {
        let type_str = type_str.as_scalar().ok_or_else(|| DTypeError::expected_scalar(type_str, "Vec<u8>"))?;
        let size = match usize::try_from(type_str.size) {
            Ok(size) => size,
            Err(_) => return Err(DTypeError::bad_usize(type_str.size)),
        };

        let is_byte_str = match *type_str {
            TypeStr { type_kind: TypeKind::ByteStr, .. } => true,
            TypeStr { type_kind: TypeKind::RawData, .. } => false,
            _ => return Err(DTypeError::bad_scalar("read", type_str, "Vec<u8>")),
        };
        Ok(BytesReader { size, is_byte_str })
    }
}


pub struct StringReader {
    bytes_reader: BytesReader,
}

impl TypeRead for StringReader {
    type Value = String;

    fn read_one<R: io::Read>(&self, bytes: R) -> io::Result<Self::Value>
    {
        let v = self.bytes_reader.read_one(bytes)?;
        match String::from_utf8(v) {
            Ok(s) => Ok(s),
            Err(e) => Err(io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Can not convert to utf-8: {}", e),
            )),
        }
    }
}

impl Deserialize for String {
    type TypeReader = StringReader;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        Ok(StringReader {
            bytes_reader: <Vec<u8> as Deserialize>::reader(dtype)?,
        })
    }
}




pub struct BytesWriter {
    type_str: TypeStr,
    size: usize,
    is_byte_str: bool,
}

impl TypeWrite for BytesWriter {
    type Value = [u8];

    fn write_one<W: io::Write>(&self, mut w: W, bytes: &[u8]) -> io::Result<()> {
        use std::cmp::Ordering;

        match (bytes.len().cmp(&self.size), self.is_byte_str) {
            (Ordering::Greater, _) |
            (Ordering::Less, false) => return invalid_data(
                &format!("bad item length {} for type-string '{}'", bytes.len(), self.type_str),
            ),
            _ => {},
        }

        w.write_all(bytes)?;
        if self.is_byte_str {
            w.write_all(&vec![0; self.size - bytes.len()])?;
        }
        Ok(())
    }
}

impl Serialize for [u8] {
    type TypeWriter = BytesWriter;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let type_str = dtype.as_scalar().ok_or_else(|| DTypeError::expected_scalar(dtype, "[u8]"))?;

        let size = match usize::try_from(type_str.size) {
            Ok(size) => size,
            Err(_) => return Err(DTypeError::bad_usize(type_str.size)),
        };

        let type_str = type_str.clone();
        let is_byte_str = match type_str {
            TypeStr { type_kind: TypeKind::ByteStr, .. } => true,
            TypeStr { type_kind: TypeKind::RawData, .. } => false,
            _ => return Err(DTypeError::bad_scalar("read", &type_str, "[u8]")),
        };
        Ok(BytesWriter { type_str, size, is_byte_str })
    }
}


#[cfg(feature = "arraystring")]
pub struct ArrayStringReader<const CAP: usize> {
    bytes_reader: BytesReader,
}

#[cfg(feature = "arraystring")]
impl<const CAP: usize> TypeRead for ArrayStringReader<CAP> {
    type Value = ArrayString<CAP>;

    fn read_one<R: io::Read>(&self, bytes: R) -> io::Result<Self::Value>
    {
        let v = self.bytes_reader.read_one(bytes)?;
        let mut data = [0_u8; CAP];
        v.iter().take(CAP).enumerate().for_each(|(i, c)| {
            data[i] = *c;
        });
        match ArrayString::<CAP>::from_byte_string(&data) {
            Ok(s) => Ok(s),
            Err(e) => Err(io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Can not convert to utf-8: {}", e),
            )),
        }
    }
}

#[cfg(feature = "arraystring")]
impl<const CAP: usize> Deserialize for ArrayString<CAP> {
    type TypeReader = ArrayStringReader<CAP>;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        Ok(ArrayStringReader::<CAP> {
            bytes_reader: <Vec<u8> as Deserialize>::reader(dtype)?,
        })
    }
}

#[cfg(feature = "arraystring")]
pub struct ArrayStringWriter<const CAP: usize> {
    bytes_writer: BytesWriter,
}

#[cfg(feature = "arraystring")]
impl<const CAP: usize> TypeWrite for ArrayStringWriter<CAP> {
    type Value = ArrayString<CAP>;

    fn write_one<W: io::Write>(&self, writer: W, value: &Self::Value) -> io::Result<()> where Self: Sized {
        let bytes = value.as_bytes();
        self.bytes_writer.write_one(writer, bytes)
    }
}

#[cfg(feature = "arraystring")]
impl<const CAP: usize> Serialize for ArrayString<CAP> {
    type TypeWriter = ArrayStringWriter<CAP>;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        Ok(ArrayStringWriter::<CAP> {
            bytes_writer: <[u8] as Serialize>::writer(dtype)?
        })
    }
}

#[cfg(feature = "arraystring")]
impl<const CAP: usize> AutoSerialize for ArrayString<CAP> {
    fn default_dtype() -> DType {
        DType::Plain(TypeStr {
            endianness: Endianness::Little,
            type_kind: TypeKind::ByteStr,
            size: CAP as u64,
            time_units: None,
        })
    }
}


#[macro_use]
mod helper {
    use super::*;
    use std::ops::Deref;

    pub struct TypeWriteViaDeref<T>
    where
        T: Deref,
        <T as Deref>::Target: Serialize,
    {
        pub(crate) inner: <<T as Deref>::Target as Serialize>::TypeWriter,
    }

    impl<T, U: ?Sized> TypeWrite for TypeWriteViaDeref<T>
    where
        T: Deref<Target=U>,
        U: Serialize,
    {
        type Value = T;

        #[inline(always)]
        fn write_one<W: io::Write>(&self, writer: W, value: &T) -> io::Result<()> {
            self.inner.write_one(writer, value)
        }
    }

    macro_rules! impl_serialize_by_deref {
        ([$($generics:tt)*] $T:ty => $Target:ty $(where $($bounds:tt)+)*) => {
            impl<$($generics)*> Serialize for $T
            $(where $($bounds)+)*
            {
                type TypeWriter = helper::TypeWriteViaDeref<$T>;

                #[inline(always)]
                fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
                    Ok(helper::TypeWriteViaDeref { inner: <$Target>::writer(dtype)? })
                }
            }
        };
    }

    macro_rules! impl_auto_serialize {
        ([$($generics:tt)*] $T:ty as $Delegate:ty $(where $($bounds:tt)+)*) => {
            impl<$($generics)*> AutoSerialize for $T
            $(where $($bounds)+)*
            {
                #[inline(always)]
                fn default_dtype() -> DType {
                    <$Delegate>::default_dtype()
                }
            }
        };
    }
}

impl_serialize_by_deref!{[] Vec<u8> => [u8]}

impl_serialize_by_deref!{['a, T: ?Sized] &'a T => T where T: Serialize}
impl_serialize_by_deref!{['a, T: ?Sized] &'a mut T => T where T: Serialize}
impl_serialize_by_deref!{[T: ?Sized] Box<T> => T where T: Serialize}
impl_serialize_by_deref!{[T: ?Sized] std::rc::Rc<T> => T where T: Serialize}
impl_serialize_by_deref!{[T: ?Sized] std::sync::Arc<T> => T where T: Serialize}
impl_serialize_by_deref!{['a, T: ?Sized] std::borrow::Cow<'a, T> => T where T: Serialize + std::borrow::ToOwned}
impl_auto_serialize!{[T: ?Sized] &T as T where T: AutoSerialize}
impl_auto_serialize!{[T: ?Sized] &mut T as T where T: AutoSerialize}
impl_auto_serialize!{[T: ?Sized] Box<T> as T where T: AutoSerialize}
impl_auto_serialize!{[T: ?Sized] std::rc::Rc<T> as T where T: AutoSerialize}
impl_auto_serialize!{[T: ?Sized] std::sync::Arc<T> as T where T: AutoSerialize}
impl_auto_serialize!{[T: ?Sized] std::borrow::Cow<'_, T> as T where T: AutoSerialize + std::borrow::ToOwned}

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

mod arrays {
    use super::*;

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
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    // NOTE: Tests for arrays are in tests/serialize_array.rs because they require derives

    fn reader_output<T: Deserialize>(dtype: &DType, bytes: &[u8]) -> T {
        T::reader(dtype).unwrap_or_else(|e| panic!("{}", e)).read_one(bytes).expect("reader_output failed")
    }

    fn reader_expect_err<T: Deserialize>(dtype: &DType) {
        T::reader(dtype).err().expect("reader_expect_err failed!");
    }

    fn writer_output<T: Serialize + ?Sized>(dtype: &DType, value: &T) -> Vec<u8> {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value).unwrap();
        vec
    }

    fn writer_expect_err<T: Serialize + ?Sized>(dtype: &DType) {
        T::writer(dtype).err().expect("writer_expect_err failed!");
    }

    fn writer_expect_write_err<T: Serialize + ?Sized>(dtype: &DType, value: &T) {
        let mut vec = vec![];
        T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
            .write_one(&mut vec, value)
            .err().expect("writer_expect_write_err failed!");
    }

    const BE_ONE_64: &[u8] = &[0, 0, 0, 0, 0, 0, 0, 1];
    const LE_ONE_64: &[u8] = &[1, 0, 0, 0, 0, 0, 0, 0];
    const BE_ONE_32: &[u8] = &[0, 0, 0, 1];
    const LE_ONE_32: &[u8] = &[1, 0, 0, 0];

    #[test]
    fn identity() {
        let be = DType::parse("'>i4'").unwrap();
        let le = DType::parse("'<i4'").unwrap();

        assert_eq!(reader_output::<i32>(&be, BE_ONE_32), 1);
        assert_eq!(reader_output::<i32>(&le, LE_ONE_32), 1);
        assert_eq!(writer_output::<i32>(&be, &1), BE_ONE_32);
        assert_eq!(writer_output::<i32>(&le, &1), LE_ONE_32);

        let be = DType::parse("'>u4'").unwrap();
        let le = DType::parse("'<u4'").unwrap();

        assert_eq!(reader_output::<u32>(&be, BE_ONE_32), 1);
        assert_eq!(reader_output::<u32>(&le, LE_ONE_32), 1);
        assert_eq!(writer_output::<u32>(&be, &1), BE_ONE_32);
        assert_eq!(writer_output::<u32>(&le, &1), LE_ONE_32);

        for &dtype in &["'>i1'", "'<i1'", "'|i1'"] {
            let dtype = DType::parse(dtype).unwrap();
            assert_eq!(reader_output::<i8>(&dtype, &[1]), 1);
            assert_eq!(writer_output::<i8>(&dtype, &1), &[1][..]);
        }

        for &dtype in &["'>u1'", "'<u1'", "'|u1'"] {
            let dtype = DType::parse(dtype).unwrap();
            assert_eq!(reader_output::<u8>(&dtype, &[1]), 1);
            assert_eq!(writer_output::<u8>(&dtype, &1), &[1][..]);
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
    #[cfg(feature = "complex")]
    fn native_complex_types() {
        use num_complex::{Complex32, Complex64};

        let c = Complex64 { re: 42.0, im: 63.0 };

        let mut be_bytes = [0; 16];
        be_bytes[..8].copy_from_slice(&c.re.to_bits().to_be_bytes());
        be_bytes[8..].copy_from_slice(&c.im.to_bits().to_be_bytes());

        let mut le_bytes = [0; 16];
        le_bytes[..8].copy_from_slice(&c.re.to_bits().to_le_bytes());
        le_bytes[8..].copy_from_slice(&c.im.to_bits().to_le_bytes());

        let be = DType::parse(&format!("'>c16'")).unwrap();
        let le = DType::parse(&format!("'<c16'")).unwrap();

        assert_eq!(reader_output::<Complex64>(&be, &be_bytes), c);
        assert_eq!(reader_output::<Complex64>(&le, &le_bytes), c);
        assert_eq!(writer_output::<Complex64>(&be, &c), &be_bytes);
        assert_eq!(writer_output::<Complex64>(&le, &c), &le_bytes);

        let c = Complex32 { re: 42.0, im: 63.0 };

        let mut be_bytes = [0; 8];
        be_bytes[..4].copy_from_slice(&c.re.to_bits().to_be_bytes());
        be_bytes[4..].copy_from_slice(&c.im.to_bits().to_be_bytes());

        let mut le_bytes = [0; 8];
        le_bytes[..4].copy_from_slice(&c.re.to_bits().to_le_bytes());
        le_bytes[4..].copy_from_slice(&c.im.to_bits().to_le_bytes());

        let be = DType::parse(&format!("'>c8'")).unwrap();
        let le = DType::parse(&format!("'<c8'")).unwrap();

        assert_eq!(reader_output::<Complex32>(&be, &be_bytes), c);
        assert_eq!(reader_output::<Complex32>(&le, &le_bytes), c);
        assert_eq!(writer_output::<Complex32>(&be, &c), &be_bytes);
        assert_eq!(writer_output::<Complex32>(&le, &c), &le_bytes);
    }

    #[test]
    fn datetime_as_int() {
        let be = DType::parse("'>m8[ns]'").unwrap();
        let le = DType::parse("'<m8[ns]'").unwrap();

        assert_eq!(reader_output::<i64>(&be, BE_ONE_64), 1);
        assert_eq!(reader_output::<i64>(&le, LE_ONE_64), 1);
        assert_eq!(writer_output::<i64>(&be, &1), BE_ONE_64);
        assert_eq!(writer_output::<i64>(&le, &1), LE_ONE_64);

        let be = DType::parse("'>M8[ns]'").unwrap();
        let le = DType::parse("'<M8[ns]'").unwrap();

        assert_eq!(reader_output::<u64>(&be, BE_ONE_64), 1);
        assert_eq!(reader_output::<u64>(&le, LE_ONE_64), 1);
        assert_eq!(writer_output::<u64>(&be, &1), BE_ONE_64);
        assert_eq!(writer_output::<u64>(&le, &1), LE_ONE_64);
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
    fn bytes_any_endianness() {
        for ty in vec!["'<S3'", "'>S3'", "'|S3'"] {
            let ty = DType::parse(ty).unwrap();
            assert_eq!(writer_output(&ty, &[1, 3, 5][..]), vec![1, 3, 5]);
            assert_eq!(reader_output::<Vec<u8>>(&ty, &[1, 3, 5][..]), vec![1, 3, 5]);
        }
    }

    #[test]
    fn bytes_size_zero() {
        let ts = DType::parse("'|S0'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[]), vec![]);
        assert_eq!(writer_output(&ts, &[][..]), vec![]);

        let ts = DType::parse("'|V0'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[]), vec![]);
        assert_eq!(writer_output::<[u8]>(&ts, &[]), vec![]);
    }

    #[test]
    fn wrong_size_bytes() {
        let s_3 = DType::parse("'|S3'").unwrap();
        let v_3 = DType::parse("'|V3'").unwrap();

        assert_eq!(writer_output(&s_3, &[1, 3, 5][..]), vec![1, 3, 5]);
        assert_eq!(writer_output(&v_3, &[1, 3, 5][..]), vec![1, 3, 5]);

        assert_eq!(writer_output(&s_3, &[1][..]), vec![1, 0, 0]);
        writer_expect_write_err(&v_3, &[1][..]);

        assert_eq!(writer_output(&s_3, &[][..]), vec![0, 0, 0]);
        writer_expect_write_err(&v_3, &[][..]);

        writer_expect_write_err(&s_3, &[1, 3, 5, 7][..]);
        writer_expect_write_err(&v_3, &[1, 3, 5, 7][..]);
    }

    #[test]
    fn read_bytes_with_trailing_zeros() {
        let ts = DType::parse("'|S2'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[1, 3]), vec![1, 3]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[1, 0]), vec![1]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[0, 0]), vec![]);

        let ts = DType::parse("'|V2'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[1, 3]), vec![1, 3]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[1, 0]), vec![1, 0]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[0, 0]), vec![0, 0]);
    }

    #[test]
    fn bytestr_preserves_interior_zeros() {
        const DATA: &[u8] = &[0, 1, 0, 0, 3, 5];

        let ts = DType::parse("'|S6'").unwrap();

        assert_eq!(reader_output::<Vec<u8>>(&ts, DATA), DATA.to_vec());
        assert_eq!(writer_output(&ts, DATA), DATA.to_vec());
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
    fn serialize_types_that_deref_to_bytes() {
        let ts = DType::parse("'|S3'").unwrap();

        assert_eq!(writer_output::<Vec<u8>>(&ts, &vec![1, 3, 5]), vec![1, 3, 5]);
        assert_eq!(writer_output::<&[u8]>(&ts, &&[1, 3, 5][..]), vec![1, 3, 5]);
    }

    #[test]
    fn dynamic_readers_and_writers() {
        let writer: Box<dyn TypeWriteDyn<Value=i32>> = Box::new(i32::writer(&i32::default_dtype()).unwrap());
        let reader: Box<dyn TypeReadDyn<Value=i32>> = Box::new(i32::reader(&i32::default_dtype()).unwrap());

        let mut buf = vec![];
        writer.write_one(&mut buf, &4000).unwrap();
        assert_eq!(reader.read_one(&buf[..]).unwrap(), 4000);
    }
}
