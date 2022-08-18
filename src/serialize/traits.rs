use std::io;
use std::fmt;

use crate::header::DType;
use crate::type_str::{TypeStr};

#[allow(unused)] // used by docstrings
use crate::type_matchup_docs;

/// Trait that permits reading a type from an `.npy` file.
///
/// > Complete documentation of all types that implement this trait is available on the
/// > [`type_matchup_docs`] module.
///
/// Examples of types that implement this:
///
/// * Primitive integers, floats, `Complex` (with the **`"complex"`** feature)
/// * Owned containers (`Vec<u8>`, `String`)
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
/// > Complete documentation of all types that implement this trait is available on the
/// > [`type_matchup_docs`] module.
///
/// Examples of types that implement this:
///
/// * Primitive integers, floats, `Complex` (with the **`"complex"`** feature)
/// * Slice types (`[u8]`, `str`)
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
/// > Complete documentation of all types that implement this trait is available on the
/// > [`type_matchup_docs`] module.
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
/// To obtain one of these, use the [`Deserialize`] trait.
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
    fn read_one<R: io::Read>(&self, bytes: R) -> io::Result<Self::Value>
        where Self: Sized;
}

/// Like some sort of `for<W: io::Write> Fn(W, &T) -> io::Result<()>`.
///
/// To obtain one of these, use the [`Serialize`] trait.
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

impl<T> TypeRead for Box<dyn TypeReadDyn<Value=T>> {
    type Value = T;

    #[inline(always)]
    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<T> where Self: Sized {
        (**self).read_one_dyn(&mut reader)
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

// =============================================================================
// Error type

/// Indicates that a particular rust type does not support serialization or deserialization
/// as a given [`DType`].
#[derive(Debug, Clone)]
pub struct DTypeError(pub(crate) ErrorKind);

#[derive(Debug, Clone)]
pub(crate) enum ErrorKind {
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
    pub fn custom<S: core::fmt::Display>(msg: S) -> Self {
        DTypeError(ErrorKind::Custom(msg.to_string()))
    }

    // verb should be "read" or "write"
    pub(crate) fn bad_scalar<T: ?Sized>(verb: &'static str, type_str: &TypeStr) -> Self {
        let type_str = type_str.clone();
        let rust_type = std::any::type_name::<T>();
        DTypeError(ErrorKind::BadScalar { type_str, rust_type, verb })
    }

    pub(crate) fn bad_usize(x: u64) -> Self {
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

// =============================================================================
// Generic/forwarded impls

#[macro_use]
pub(in crate::serialize) mod helper {
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
        (
            $(#[$($attr:tt)+])*
            [$($generics:tt)*] $T:ty => $Target:ty $(where $($bounds:tt)+)*
        ) => {
            $(#[$($attr)+])*
            impl<$($generics)*> Serialize for $T
            $(where $($bounds)+)*
            {
                type TypeWriter = crate::serialize::helper::TypeWriteViaDeref<$T>;

                #[inline(always)]
                fn writer(dtype: &DType) -> core::result::Result<Self::TypeWriter, DTypeError> {
                    Ok(crate::serialize::helper::TypeWriteViaDeref { inner: <$Target>::writer(dtype)? })
                }
            }
        };
    }

    macro_rules! impl_auto_serialize {
        ([$($generics:tt)*] $T:ty as $Delegate:ty $(where $($bounds:tt)+)*) => {
            impl<$($generics)*> crate::serialize::AutoSerialize for $T
            $(where $($bounds)+)*
            {
                #[inline(always)]
                fn default_dtype() -> crate::DType {
                    <$Delegate>::default_dtype()
                }
            }
        };
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_readers_and_writers() {
        let writer: Box<dyn TypeWriteDyn<Value=i32>> = Box::new(i32::writer(&i32::default_dtype()).unwrap());
        let reader: Box<dyn TypeReadDyn<Value=i32>> = Box::new(i32::reader(&i32::default_dtype()).unwrap());

        let mut buf = vec![];
        writer.write_one(&mut buf, &4000).unwrap();
        assert_eq!(reader.read_one(&buf[..]).unwrap(), 4000);
    }
}