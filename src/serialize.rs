use header::DType;
use type_str::{TypeStr, Endianness, TypeKind};
use byteorder::{ByteOrder, NativeEndian, WriteBytesExt};
use self::{TypeKind::*};
use std::io;
use std::fmt;
use std::convert::TryFrom;

/// Trait that permits reading a type from an `.npy` file.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait Deserialize: Sized {
    /// Think of this as like a `Fn(&[u8]) -> (Self, &[u8])`.
    ///
    /// There is no closure-like sugar for these; you must manually define a type that
    /// implements [`TypeRead`].
    type Reader: TypeRead<Value=Self>;

    /// Get a function that deserializes a single data field at a time
    ///
    /// The function receives a byte buffer containing at least
    /// `dtype.num_bytes()` bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `DType` is not compatible with `Self`.
    fn reader(dtype: &DType) -> Result<Self::Reader, DTypeError>;
}

/// Trait that permits writing a type to an `.npy` file.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait Serialize {
    /// Think of this as some sort of `for<W: io::Write> Fn(W, &Self) -> io::Result<()>`.
    ///
    /// There is no closure-like sugar for these; you must manually define a type that
    /// implements [`TypeWrite`].
    type Writer: TypeWrite<Value=Self>;

    /// Get a function that serializes a single data field at a time.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the `DType` is not compatible with `Self`.
    fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError>;
}

/// Subtrait of [`Serialize`] for types which have a reasonable default [`DType`].
///
/// This opens up some simpler APIs for serialization. (e.g. [`::to_file`])
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait AutoSerialize: Serialize {
    /// A suggested format for serialization.
    ///
    /// The builtin implementations for primitive types generally prefer `|` endianness if possible,
    /// else the machine endian format.
    fn default_dtype() -> DType;
}

/// Like a `Fn(&[u8]) -> (T, &[u8])`.
///
/// It is a separate trait from `Fn` for consistency with [`TypeWrite`], and so that
/// default methods can potentially be added in the future that may be overriden
/// for efficiency.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait TypeRead {
    /// Type returned by the function.
    type Value;

    /// The function.
    ///
    /// Receives *at least* enough bytes to read `Self::Value`, and returns the remainder.
    fn read_one<'a>(&self, bytes: &'a [u8]) -> (Self::Value, &'a [u8]);
}

/// Like some sort of `for<W: io::Write> Fn(W, &T) -> io::Result<()>`.
///
/// For an example of how to implement this, please see the
/// [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).
pub trait TypeWrite {
    /// Type accepted by the function.
    type Value: ?Sized;

    /// The function.
    fn write_one<W: io::Write>(&self, writer: W, value: &Self::Value) -> io::Result<()>
    where Self: Sized;
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
        type_str: TypeStr,
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
    pub fn expected_record(type_str: &TypeStr) -> Self {
        let type_str = type_str.clone();
        DTypeError(ErrorKind::ExpectedRecord { type_str })
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
            ErrorKind::ExpectedRecord { type_str } => {
                write!(f, "expected a record type; got a scalar type '{}'", type_str)
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

impl<T> TypeRead for Box<dyn TypeRead<Value=T>> {
    type Value = T;

    #[inline(always)]
    fn read_one<'a>(&self, bytes: &'a [u8]) -> (T, &'a [u8]) {
        (**self).read_one(bytes)
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

fn invalid_data<T>(message: &str) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message.to_string()))
}

// Takes info about each data size, from largest to smallest.
macro_rules! impl_integer_serializable {
    ( @iterate
        meta: $meta:tt
        remaining: []
    ) => {};

    ( @iterate
        meta: $meta:tt
        remaining: [$first:tt $($smaller:tt)*]
    ) => {
        impl_integer_serializable! {
          @generate
            meta: $meta
            current: $first
        }

        impl_integer_serializable! {
          @iterate
            meta: $meta
            remaining: [ $($smaller)* ]
        }
    };

    (
        @generate
        meta: [ (main_ty: $Int:ident) (date_ty: $DateTime:ident) ]
        current: [ $size:literal $int:ident
                   (size1: $size1_cfg:meta) $read_int:ident $write_int:ident
                 ]
    ) => {
        mod $int {
            use super::*;

            pub struct AnyEndianReader { pub(super) swap_byteorder: bool }
            pub struct AnyEndianWriter { pub(super) swap_byteorder: bool }

            pub(super) fn expect_scalar_dtype(dtype: &DType) -> Result<&TypeStr, DTypeError> {
                dtype.as_scalar().ok_or_else(|| {
                    DTypeError::expected_scalar(dtype, stringify!($int))
                })
            }

            #[inline]
            fn maybe_swap(swap: bool, x: $int) -> $int {
                match swap {
                    true => x.to_be().to_le(),
                    false => x,
                }
            }

            impl TypeRead for AnyEndianReader {
                type Value = $int;

                #[inline(always)]
                fn read_one<'a>(&self, bytes: &'a [u8]) -> (Self::Value, &'a [u8]) {
                    let value = maybe_swap(self.swap_byteorder, NativeEndian::$read_int(bytes));
                    (value, &bytes[$size..])
                }
            }

            impl TypeWrite for AnyEndianWriter {
                type Value = $int;

                #[inline(always)]
                fn write_one<W: io::Write>(&self, mut writer: W, &value: &Self::Value) -> io::Result<()> {
                    writer.$write_int::<NativeEndian>(maybe_swap(self.swap_byteorder, value))
                }
            }
        }

        impl Deserialize for $int {
            type Reader = $int::AnyEndianReader;

            fn reader(dtype: &DType) -> Result<Self::Reader, DTypeError> {
                match $int::expect_scalar_dtype(dtype)? {
                    // Read an integer of the same size and signedness.
                    //
                    // DateTime is an unsigned integer and TimeDelta is a signed integer,
                    // so we support those too.
                    TypeStr { size: $size, endianness, type_kind: $Int, .. } |
                    TypeStr { size: $size, endianness, type_kind: $DateTime, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok($int::AnyEndianReader { swap_byteorder })
                    },
                    type_str => Err(DTypeError::bad_scalar("read", type_str, stringify!($int))),
                }
            }
        }

        impl Serialize for $int {
            type Writer = $int::AnyEndianWriter;

            fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError> {
                match $int::expect_scalar_dtype(dtype)? {
                    // Write a signed integer of the correct size
                    TypeStr { size: $size, endianness, type_kind: $Int, .. } |
                    TypeStr { size: $size, endianness, type_kind: $DateTime, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok($int::AnyEndianWriter { swap_byteorder })
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
    };
}

// Needed by the macro: Methods missing from byteorder
trait ReadSingleByteExt {
    #[inline(always)] fn read_u8_(bytes: &[u8]) -> u8 { bytes[0] }
    #[inline(always)] fn read_i8_(bytes: &[u8]) -> i8 { i8::from_ne_bytes([bytes[0]]) }
}

impl<E: ByteOrder> ReadSingleByteExt for E {}

/// Needed by the macro: Methods modified to take a generic type param
trait WriteSingleByteExt: WriteBytesExt {
    #[inline(always)] fn write_u8_<B>(&mut self, value: u8) -> io::Result<()> { self.write_u8(value) }
    #[inline(always)] fn write_i8_<B>(&mut self, value: i8) -> io::Result<()> { self.write_i8(value) }
}

impl<W: WriteBytesExt + ?Sized> WriteSingleByteExt for W {}

// `all()` means "true", `any()` means "false". (these get put inside `cfg`)
impl_integer_serializable! {
    @iterate
    meta: [ (main_ty: Int) (date_ty: TimeDelta) ]
    remaining: [
        // numpy doesn't support i128
        [ 8  i64 (size1: any()) read_i64  write_i64 ]
        [ 4  i32 (size1: any()) read_i32  write_i32 ]
        [ 2  i16 (size1: any()) read_i16  write_i16 ]
        [ 1   i8 (size1: all()) read_i8_  write_i8_ ]
    ]
}

impl_integer_serializable! {
    @iterate
    meta: [ (main_ty: Uint) (date_ty: DateTime) ]
    remaining: [
        // numpy doesn't support i128
        [ 8  u64 (size1: any()) read_u64  write_u64 ]
        [ 4  u32 (size1: any()) read_u32  write_u32 ]
        [ 2  u16 (size1: any()) read_u16  write_u16 ]
        [ 1   u8 (size1: all()) read_u8_  write_u8_ ]
    ]
}

// Takes info about each data size, from largest to smallest.
macro_rules! impl_float_serializable {
    ( $( [ $size:literal $float:ident $read_float:ident $write_float:ident ] )+ ) => { $(
        mod $float {
            use super::*;

            pub struct AnyEndianReader { pub(super) swap_byteorder: bool }
            pub struct AnyEndianWriter { pub(super) swap_byteorder: bool }

            #[inline]
            fn maybe_swap(swap: bool, x: $float) -> $float {
                match swap {
                    true => $float::from_bits(x.to_bits().to_be().to_le()),
                    false => x,
                }
            }

            pub(super) fn expect_scalar_dtype(dtype: &DType) -> Result<&TypeStr, DTypeError> {
                dtype.as_scalar().ok_or_else(|| {
                    DTypeError::expected_scalar(dtype, stringify!($float))
                })
            }

            impl TypeRead for AnyEndianReader {
                type Value = $float;

                #[inline(always)]
                fn read_one<'a>(&self, bytes: &'a [u8]) -> ($float, &'a [u8]) {
                    let value = maybe_swap(self.swap_byteorder, NativeEndian::$read_float(bytes));
                    (value, &bytes[$size..])
                }
            }

            impl TypeWrite for AnyEndianWriter {
                type Value = $float;

                #[inline(always)]
                fn write_one<W: io::Write>(&self, mut writer: W, &value: &$float) -> io::Result<()> {
                    writer.$write_float::<NativeEndian>(maybe_swap(self.swap_byteorder, value))
                }
            }
        }

        impl Deserialize for $float {
            type Reader = $float::AnyEndianReader;

            fn reader(dtype: &DType) -> Result<Self::Reader, DTypeError> {
                match $float::expect_scalar_dtype(dtype)? {
                    // Read a float of the correct size
                    TypeStr { size: $size, endianness, type_kind: Float, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok($float::AnyEndianReader { swap_byteorder })
                    },
                    type_str => Err(DTypeError::bad_scalar("read", type_str, stringify!($float))),
                }
            }
        }

        impl Serialize for $float {
            type Writer = $float::AnyEndianWriter;

            fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError> {
                match $float::expect_scalar_dtype(dtype)? {
                    // Write a float of the correct size
                    TypeStr { size: $size, endianness, type_kind: Float, .. } => {
                        let swap_byteorder = endianness.requires_swap(Endianness::of_machine());
                        Ok($float::AnyEndianWriter { swap_byteorder })
                    },
                    type_str => Err(DTypeError::bad_scalar("write", type_str, stringify!($float))),
                }
            }
        }

        impl AutoSerialize for $float {
            fn default_dtype() -> DType {
                DType::new_scalar(TypeStr::with_auto_endianness(Float, $size, None))
            }
        }
    )+};
}

impl_float_serializable! {
    // TODO: numpy supports f16, f128
    [ 8 f64 read_f64 write_f64 ]
    [ 4 f32 read_f32 write_f32 ]
}

pub struct BytesReader {
    size: usize,
    is_byte_str: bool,
}

impl TypeRead for BytesReader {
    type Value = Vec<u8>;

    fn read_one<'a>(&self, bytes: &'a [u8]) -> (Vec<u8>, &'a [u8]) {
        let mut vec = vec![];

        let (src, remainder) = bytes.split_at(self.size);
        vec.resize(self.size, 0);
        vec.copy_from_slice(src);

        // truncate trailing zeros for type 'S'
        if self.is_byte_str {
            let end = vec.iter().rposition(|x| x != &0).map_or(0, |ind| ind + 1);
            vec.truncate(end);
        }

        (vec, remainder)
    }
}

impl Deserialize for Vec<u8> {
    type Reader = BytesReader;

    fn reader(type_str: &DType) -> Result<Self::Reader, DTypeError> {
        let type_str = type_str.as_scalar().ok_or_else(|| DTypeError::expected_scalar(type_str, "Vec<u8>"))?;
        let size = match usize::try_from(type_str.size) {
            Ok(size) => size,
            Err(_) => return Err(DTypeError::bad_usize(type_str.size)),
        };

        let is_byte_str = match *type_str {
            TypeStr { type_kind: ByteStr, .. } => true,
            TypeStr { type_kind: RawData, .. } => false,
            _ => return Err(DTypeError::bad_scalar("read", type_str, "Vec<u8>")),
        };
        Ok(BytesReader { size, is_byte_str })
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
    type Writer = BytesWriter;

    fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError> {
        let type_str = dtype.as_scalar().ok_or_else(|| DTypeError::expected_scalar(dtype, "[u8]"))?;

        let size = match usize::try_from(type_str.size) {
            Ok(size) => size,
            Err(_) => return Err(DTypeError::bad_usize(type_str.size)),
        };

        let type_str = type_str.clone();
        let is_byte_str = match type_str {
            TypeStr { type_kind: ByteStr, .. } => true,
            TypeStr { type_kind: RawData, .. } => false,
            _ => return Err(DTypeError::bad_scalar("read", &type_str, "[u8]")),
        };
        Ok(BytesWriter { type_str, size, is_byte_str })
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
        pub(crate) inner: <<T as Deref>::Target as Serialize>::Writer,
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
                type Writer = helper::TypeWriteViaDeref<$T>;

                #[inline(always)]
                fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError> {
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
    /// Expect an array dtype, get the length of the array and the inner dtype.
    fn array_inner_dtype(&self, expected_len: u64) -> Result<Self, DTypeError> {
        match *self {
            DType::Record { .. } => Err(DTypeError(ErrorKind::ExpectedArray { got: "a record" })),
            DType::Plain { ref ty, ref shape } => {
                let ty = ty.clone();
                let mut shape = shape.to_vec();

                let len = match shape.is_empty() {
                    true => return Err(DTypeError(ErrorKind::ExpectedArray { got: "a scalar" })),
                    false => shape.remove(0),
                };

                if len != expected_len {
                    return Err(DTypeError(ErrorKind::WrongArrayLen {
                        actual: len,
                        expected: expected_len,
                    }));
                }

                Ok(DType::Plain { ty, shape })
            },
        }
    }
}

macro_rules! gen_array_serializable {
    ($([$n:tt in mod $mod_name:ident])+) => { $(
        mod $mod_name {
            use super::*;

            pub struct ArrayReader<I>{ inner: I }
            pub struct ArrayWriter<I>{ inner: I }

            impl<I: TypeRead> TypeRead for ArrayReader<I>
            where I::Value: Copy + Default,
            {
                type Value = [I::Value; $n];

                #[inline]
                fn read_one<'a>(&self, bytes: &'a [u8]) -> (Self::Value, &'a [u8]) {
                    let mut value = [I::Value::default(); $n];

                    let mut remainder = bytes;
                    for place in &mut value {
                        let (item, new_remainder) = self.inner.read_one(remainder);
                        *place = item;
                        remainder = new_remainder;
                    }

                    (value, remainder)
                }
            }

            impl<I: TypeWrite> TypeWrite for ArrayWriter<I>
            where I::Value: Sized,
            {
                type Value = [I::Value; $n];

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

            impl<T: AutoSerialize + Default + Copy> AutoSerialize for [T; $n] {
                #[inline]
                fn default_dtype() -> DType {
                    use DType::*;

                    match T::default_dtype() {
                        Plain { ty, mut shape } => DType::Plain {
                            ty,
                            shape: { shape.insert(0, $n); shape },
                        },
                        Record(_) => unimplemented!("arrays of nested records")
                    }
                }
            }

            impl<T: Deserialize + Default + Copy> Deserialize for [T; $n] {
                type Reader = ArrayReader<<T as Deserialize>::Reader>;

                #[inline]
                fn reader(dtype: &DType) -> Result<Self::Reader, DTypeError> {
                    let inner_dtype = dtype.array_inner_dtype($n)?;
                    let inner = <T>::reader(&inner_dtype)?;
                    Ok(ArrayReader { inner })
                }
            }

            impl<T: Serialize> Serialize for [T; $n] {
                type Writer = ArrayWriter<<T as Serialize>::Writer>;

                #[inline]
                fn writer(dtype: &DType) -> Result<Self::Writer, DTypeError> {
                    let inner = <T>::writer(&dtype.array_inner_dtype($n)?)?;
                    Ok(ArrayWriter { inner })
                }
            }
        }
    )+ }
}

gen_array_serializable!{
    /*  no size 0  */ [ 1 in mod  arr1] [ 2 in mod  arr2] [ 3 in mod  arr3]
    [ 4 in mod  arr4] [ 5 in mod  arr5] [ 6 in mod  arr6] [ 7 in mod  arr7]
    [ 8 in mod  arr8] [ 9 in mod  arr9] [10 in mod arr10] [11 in mod arr11]
    [12 in mod arr12] [13 in mod arr13] [14 in mod arr14] [15 in mod arr15]
    [16 in mod arr16]
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    // NOTE: Tests for arrays are in tests/serialize_array.rs because they require derives

    fn reader_output<T: Deserialize>(dtype: &DType, bytes: &[u8]) -> T {
        T::reader(dtype).unwrap_or_else(|e| panic!("{}", e)).read_one(bytes).0
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
        let reader: Box<dyn TypeRead<Value=i32>> = Box::new(i32::reader(&i32::default_dtype()).unwrap());

        let mut buf = vec![];
        writer.write_one(&mut buf, &4000).unwrap();
        assert_eq!(reader.read_one(&buf).0, 4000);
    }
}
