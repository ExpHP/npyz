//! Vec, String, slice, str.

use std::io;
use std::convert::TryFrom;

#[cfg(feature = "arrayvec")]
use arrayvec::{ArrayVec, ArrayString};

use crate::header::DType;
use crate::type_str::{TypeStr, TypeChar};
use super::{DTypeError, TypeRead, TypeWrite, Serialize, Deserialize};
use super::primitive::{PrimitiveReader, PrimitiveWriter};
use super::{invalid_data, expect_scalar_dtype};

#[doc(hidden)]
pub struct BytesReader {
    info: ByteVecDTypeInfo,
}

impl TypeRead for BytesReader {
    type Value = Vec<u8>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Vec<u8>> {
        let mut vec = vec![0; self.info.size];
        reader.read_exact(&mut vec)?;

        if self.info.is_byte_str {
            truncate_trailing_nuls(&mut vec, |&x| x == 0);
        }

        Ok(vec)
    }
}

impl Deserialize for Vec<u8> {
    type TypeReader = BytesReader;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let info = check_byte_vec_dtype::<Self>("read", dtype)?;
        Ok(BytesReader { info })
    }
}

#[doc(hidden)]
pub struct BytesWriter {
    info: ByteVecDTypeInfo,
}

impl TypeWrite for BytesWriter {
    type Value = [u8];

    fn write_one<W: io::Write>(&self, mut w: W, bytes: &[u8]) -> io::Result<()> {
        use std::cmp::Ordering;

        match (bytes.len().cmp(&self.info.size), self.info.is_byte_str) {
            (Ordering::Greater, _) |
            (Ordering::Less, false) => return Err(bad_length(bytes, &self.info.type_str)),
            _ => {},
        }

        w.write_all(bytes)?;
        if self.info.is_byte_str {
            w.write_all(&vec![0; self.info.size - bytes.len()])?;
        }
        Ok(())
    }
}

impl Serialize for [u8] {
    type TypeWriter = BytesWriter;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let info = check_byte_vec_dtype::<Self>("write", dtype)?;
        Ok(BytesWriter { info })
    }
}

#[cfg(feature = "arrayvec")]
#[doc(hidden)]
#[non_exhaustive]
pub struct ArrayVecBytesReader<const N: usize> {
    info: ByteVecDTypeInfo,
}

#[cfg(feature = "arrayvec")]
impl<const N: usize> TypeRead for ArrayVecBytesReader<N> {
    type Value = ArrayVec<u8, N>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<ArrayVec<u8, N>> {
        let mut buffer = [0; N];
        reader.read_exact(&mut buffer[..usize::min(N, self.info.size)])?;
        if self.info.size > N {
            reader.read_exact(&mut vec![0; self.info.size - N])?;
        }

        let mut array_vec: ArrayVec<u8, N> = buffer.into();
        if self.info.is_byte_str {
            arrayvec_truncate_trailing_nuls(&mut array_vec, |&x| x == 0);
        }

        Ok(array_vec)
    }
}

/// _This impl is only available with the **`arrayvec`** feature._
#[cfg(feature = "arrayvec")]
impl<const N: usize> Deserialize for ArrayVec<u8, N> {
    type TypeReader = ArrayVecBytesReader<N>;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let info = check_array_byte_vec_dtype::<Self, N>("read", dtype)?;
        Ok(ArrayVecBytesReader { info })
    }
}

#[doc(hidden)]
#[cfg(feature = "arrayvec")]
pub struct ArrayVecBytesWriter<const N: usize> {
    bytes_writer: BytesWriter,
}

#[cfg(feature = "arrayvec")]
impl<const N: usize> TypeWrite for ArrayVecBytesWriter<N> {
    type Value = ArrayVec<u8, N>;

    fn write_one<W: io::Write>(&self, w: W, bytes: &ArrayVec<u8, N>) -> io::Result<()> {
        self.bytes_writer.write_one(w, bytes)
    }
}

/// _This impl is only available with the **`arrayvec`** feature._
#[cfg(feature = "arrayvec")]
impl<const N: usize> Serialize for ArrayVec<u8, N> {
    type TypeWriter = ArrayVecBytesWriter<N>;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let info = check_array_byte_vec_dtype::<Self, N>("write", dtype)?;
        Ok(ArrayVecBytesWriter { bytes_writer: BytesWriter { info } })
    }
}

struct ByteVecDTypeInfo { type_str: TypeStr, size: usize, is_byte_str: bool }
fn check_byte_vec_dtype<T: ?Sized>(verb: &'static str, dtype: &DType) -> Result<ByteVecDTypeInfo, DTypeError> {
    let type_str = expect_scalar_dtype::<T>(dtype)?;
    let size = size_field_as_usize(type_str)?;
    let is_byte_str = match *type_str {
        TypeStr { type_char: TypeChar::ByteStr, .. } => true,
        TypeStr { type_char: TypeChar::RawData, .. } => false,
        _ => return Err(DTypeError::bad_scalar::<T>(verb, type_str)),
    };
    Ok(ByteVecDTypeInfo { type_str: type_str.clone(), size, is_byte_str })
}

#[cfg(feature = "arrayvec")]
fn check_array_byte_vec_dtype<T: ?Sized, const N: usize>(verb: &'static str, dtype: &DType) -> Result<ByteVecDTypeInfo, DTypeError> {
    let info = check_byte_vec_dtype::<T>(verb, dtype)?;
    // prevent large |VN types at dtype parsing time because they are impossible to fit
    if !info.is_byte_str && N < info.size {
        return Err(DTypeError::bad_scalar::<T>(verb, &info.type_str));
    }
    Ok(info)
}

pub use fixed_size::FixedSizeBytes;
use crate::serialize::traits::ErrorKind;

mod fixed_size {
    /// Wrapper around `[u8; N]` that can serialize as `|VN`.  The size must match exactly.
    ///
    /// This wrapper needs to exist because `[u8; N]` itself already has another `Serialize` impl,
    /// due to the generic impl for `[T; N]`.  (basically, `[u8; N]` serializes as a field of type
    /// `|u1` and shape `[N]` in a structured record)
    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct FixedSizeBytes<const N: usize>(pub [u8; N]);

    impl<const N: usize> From<FixedSizeBytes<N>> for [u8; N] {
        fn from(FixedSizeBytes(bytes): FixedSizeBytes<N>) -> [u8; N] {
            bytes
        }
    }

    impl<const N: usize> From<[u8; N]> for FixedSizeBytes<N> {
        fn from(bytes: [u8; N]) -> FixedSizeBytes<N> {
            FixedSizeBytes(bytes)
        }
    }

    impl<const N: usize> core::ops::Deref for FixedSizeBytes<N> {
        type Target = [u8; N];

        fn deref(&self) -> &[u8; N] {
            &self.0
        }
    }

    impl<const N: usize> core::ops::DerefMut for FixedSizeBytes<N> {
        fn deref_mut(&mut self) -> &mut [u8; N] {
            &mut self.0
        }
    }

    impl<const N: usize> AsRef<[u8]> for FixedSizeBytes<N> {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    impl<const N: usize> AsMut<[u8]> for FixedSizeBytes<N> {
        fn as_mut(&mut self) -> &mut [u8] {
            &mut self.0
        }
    }
}

#[doc(hidden)]
#[non_exhaustive]
pub struct FixedSizeBytesReader<const N: usize> { }

impl<const N: usize> TypeRead for FixedSizeBytesReader<N> {
    type Value = FixedSizeBytes<N>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<FixedSizeBytes<N>> {
        let mut array = [0; N];
        reader.read_exact(&mut array)?;
        Ok(FixedSizeBytes(array))
    }
}

impl<const N: usize> Deserialize for FixedSizeBytes<N> {
    type TypeReader = FixedSizeBytesReader<N>;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        let size = size_field_as_usize(type_str)?;
        if (type_str.type_char, size) != (TypeChar::RawData, N) {
            return Err(DTypeError::bad_scalar::<Self>("read", &type_str));
        };
        Ok(FixedSizeBytesReader { })
    }
}

#[doc(hidden)]
#[non_exhaustive]
pub struct FixedSizeBytesWriter<const N: usize> {}

impl<const N: usize> TypeWrite for FixedSizeBytesWriter<N> {
    type Value = FixedSizeBytes<N>;

    fn write_one<W: io::Write>(&self, mut w: W, bytes: &FixedSizeBytes<N>) -> io::Result<()> {
        w.write_all(&bytes.0)
    }
}

impl<const N: usize> Serialize for FixedSizeBytes<N> {
    type TypeWriter = FixedSizeBytesWriter<N>;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        let size = size_field_as_usize(type_str)?;
        if (type_str.type_char, size) != (TypeChar::RawData, N) {
            return Err(DTypeError::bad_scalar::<Self>("write", &type_str));
        };
        Ok(FixedSizeBytesWriter { })
    }
}

/// Helper for reading codepoints of `U`.
struct CodePointReader {
    int_reader: PrimitiveReader<u32>,
}
/// Helper for reading codepoints of `U` as `char`.
struct CharReader {
    int_reader: PrimitiveReader<u32>,
}
/// Reads `U` to `Vec<u32>`, permitting surrogates.
#[doc(hidden)]
pub struct Utf32WithSurrogatesReader {
    codepoint_reader: CodePointReader,
    num_u32s: usize,
}
/// Reads `U` to `Vec<char>`.
#[doc(hidden)]
pub struct Utf32Reader {
    char_reader: CharReader,
    num_u32s: usize,
}
/// Reads `U` to `ArrayVec<u32, N>`, permitting surrogates.
#[cfg(feature = "arrayvec")]
#[doc(hidden)]
pub struct Utf32WithSurrogatesArrayVecReader<const N: usize> {
    codepoint_reader: CodePointReader,
    num_u32s_in_dtype: usize,
}
/// Reads `U` to `ArrayVec<char, N>`.
#[cfg(feature = "arrayvec")]
#[doc(hidden)]
pub struct Utf32ArrayVecReader<const N: usize> {
    char_reader: CharReader,
    num_u32s_in_dtype: usize,
}
/// Reads `U` to `String`.
#[doc(hidden)]
pub struct Utf32StringReader {
    char_reader: CharReader,
    num_u32s: usize,
}
/// Reads `S` to `String`.
#[doc(hidden)]
pub struct Utf8StringReader {
    bytes_reader: BytesReader,
}
/// Reads `S` to `ArrayString`.
#[cfg(feature = "arrayvec")]
#[doc(hidden)]
pub struct Utf8ArrayStringReader<const N: usize> {
    string_reader: Utf8StringReader,
}
#[doc(hidden)]
pub enum StringReader {
    Utf8(Utf8StringReader),
    Utf32(Utf32StringReader),
}

impl TypeRead for CodePointReader {
    type Value = u32;

    fn read_one<R: io::Read>(&self, reader: R) -> io::Result<u32> {
        self.int_reader.read_one(reader)
            .and_then(validate_type_u_code_unit)
    }
}

impl TypeRead for CharReader {
    type Value = char;

    fn read_one<R: io::Read>(&self, reader: R) -> io::Result<char> {
        let u32 = self.int_reader.read_one(reader)?;
        char::try_from(u32).map_err(|_| {
            invalid_data(format_args!("invalid UTF-32 code unit: {:x}", u32))
        })
    }
}

impl TypeRead for Utf32WithSurrogatesReader {
    type Value = Vec<u32>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Vec<u32>> {
        let mut vec = {
            (0..self.num_u32s)
                .map(|_| self.codepoint_reader.read_one(&mut reader))
                .collect::<io::Result<Vec<_>>>()?
        };
        truncate_trailing_nuls(&mut vec, |&x| x == 0);
        Ok(vec)
    }
}

impl TypeRead for Utf32Reader {
    type Value = Vec<char>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Vec<char>> {
        let mut vec = {
            (0..self.num_u32s)
                .map(|_| self.char_reader.read_one(&mut reader))
                .collect::<Result<Vec<_>, _>>()?
        };
        truncate_trailing_nuls(&mut vec, |&x| x == '\0');
        Ok(vec)
    }
}

impl TypeRead for Utf32StringReader {
    type Value = String;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<String> {
        let mut string = String::new();
        for _ in 0..self.num_u32s {
            string.push(self.char_reader.read_one(&mut reader)?);
        }
        while string.chars().next_back() == Some('\0') {
            string.pop();
        }
        Ok(string)
    }
}

#[cfg(feature = "arrayvec")]
impl<const N: usize> TypeRead for Utf32WithSurrogatesArrayVecReader<N> {
    type Value = ArrayVec<u32, N>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Self::Value> {
        let mut out = ArrayVec::new();
        for _ in 0..usize::min(self.num_u32s_in_dtype, N) {
            out.push(self.codepoint_reader.read_one(&mut reader)?);
        }
        for _ in N..self.num_u32s_in_dtype {
            self.codepoint_reader.read_one(&mut reader)?;
        }
        arrayvec_truncate_trailing_nuls(&mut out, |&x| x == 0);
        Ok(out)
    }
}

#[cfg(feature = "arrayvec")]
impl<const N: usize> TypeRead for Utf32ArrayVecReader<N> {
    type Value = ArrayVec<char, N>;

    fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Self::Value> {
        let mut out = ArrayVec::new();
        for _ in 0..usize::min(self.num_u32s_in_dtype, N) {
            out.push(self.char_reader.read_one(&mut reader)?);
        }
        for _ in N..self.num_u32s_in_dtype {
            self.char_reader.read_one(&mut reader)?;
        }
        arrayvec_truncate_trailing_nuls(&mut out, |&x| x == '\0');
        Ok(out)
    }
}

impl TypeRead for Utf8StringReader {
    type Value = String;

    fn read_one<R: io::Read>(&self, reader: R) -> io::Result<Self::Value> {
        let vec = self.bytes_reader.read_one(reader)?;
        String::from_utf8(vec).map_err(|e| {
            invalid_data(format_args!("invalid utf-8: {}", e))
        })
    }
}

impl TypeRead for StringReader {
    type Value = String;

    fn read_one<R: io::Read>(&self, reader: R) -> io::Result<Self::Value> {
        match self {
            StringReader::Utf8(imp) => imp.read_one(reader),
            StringReader::Utf32(imp) => imp.read_one(reader),
        }
    }
}

#[cfg(feature = "arrayvec")]
impl<const N: usize> TypeRead for Utf8ArrayStringReader<N> {
    type Value = ArrayString<N>;

    fn read_one<R: io::Read>(&self, reader: R) -> io::Result<Self::Value> {
        let mut string = self.string_reader.read_one(reader)?;
        while N < string.len() {
            string.pop();
        }
        Ok(ArrayString::try_from(&string[..]).unwrap())
    }
}

impl Utf8StringReader {
    fn try_from_dtype(dtype: &DType) -> Option<Result<Self, DTypeError>> {
        match <Vec<u8>>::reader(dtype) {
            Err(e @ DTypeError(ErrorKind::UsizeOverflow { .. })) => Some(Err(e)),
            Err(_) => None,
            Ok(bytes_reader) => {
                if !bytes_reader.info.is_byte_str {
                    return None;
                }
                Some(Ok(Utf8StringReader { bytes_reader }))
            },
        }
    }
}

impl Utf32StringReader {
    fn try_from_dtype(dtype: &DType) -> Option<Result<Self, DTypeError>> {
        let type_str = expect_scalar_dtype::<Self>(dtype).ok()?;
        if type_str.type_char != TypeChar::UnicodeStr {
            return None;
        }
        let num_u32s = match size_field_as_usize(type_str) {
            Err(e) => return Some(Err(e)),
            Ok(val) => val,
        };
        let char_reader = CharReader { int_reader: PrimitiveReader::new(type_str.endianness) };
        Some(Ok(Utf32StringReader { num_u32s, char_reader }))
    }
}

impl Deserialize for Vec<u32> {
    type TypeReader = Utf32WithSurrogatesReader;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        if type_str.type_char != TypeChar::UnicodeStr {
            return Err(DTypeError::bad_scalar::<Self>("read", &type_str));
        };

        let num_u32s = size_field_as_usize(type_str)?;
        let codepoint_reader = CodePointReader { int_reader: PrimitiveReader::new(type_str.endianness) };
        Ok(Utf32WithSurrogatesReader { num_u32s, codepoint_reader })
    }
}

impl Deserialize for Vec<char> {
    type TypeReader = Utf32Reader;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        if type_str.type_char != TypeChar::UnicodeStr {
            return Err(DTypeError::bad_scalar::<Self>("read", &type_str));
        };

        let num_u32s = size_field_as_usize(type_str)?;
        let char_reader = CharReader { int_reader: PrimitiveReader::new(type_str.endianness) };
        Ok(Utf32Reader { num_u32s, char_reader })
    }
}

impl Deserialize for String {
    type TypeReader = StringReader;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        // dispatch based on 'U' vs 'S'
        if let Some(imp) = Utf32StringReader::try_from_dtype(dtype) {
            Ok(StringReader::Utf32(imp?))
        } else if let Some(imp) = Utf8StringReader::try_from_dtype(dtype) {
            Ok(StringReader::Utf8(imp?))
        } else {
            let type_str = expect_scalar_dtype::<Self>(dtype)?;
            Err(DTypeError::bad_scalar::<Self>("read", &type_str))
        }
    }
}

/// _This impl is only available with the **`"arrayvec"`** feature._
#[cfg(feature = "arrayvec")]
impl<const N: usize> Deserialize for ArrayVec<u32, N> {
    type TypeReader = Utf32WithSurrogatesArrayVecReader<N>;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        let num_u32s_in_dtype = size_field_as_usize(type_str)?;

        if type_str.type_char != TypeChar::UnicodeStr {
            return Err(DTypeError::bad_scalar::<Self>("read", &type_str));
        };

        let codepoint_reader = CodePointReader { int_reader: PrimitiveReader::new(type_str.endianness) };
        Ok(Utf32WithSurrogatesArrayVecReader { num_u32s_in_dtype, codepoint_reader })
    }
}

/// _This impl is only available with the **`"arrayvec"`** feature._
#[cfg(feature = "arrayvec")]
impl<const N: usize> Deserialize for ArrayVec<char, N> {
    type TypeReader = Utf32ArrayVecReader<N>;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        let num_u32s_in_dtype = size_field_as_usize(type_str)?;

        if type_str.type_char != TypeChar::UnicodeStr {
            return Err(DTypeError::bad_scalar::<Self>("read", &type_str));
        };

        let char_reader = CharReader { int_reader: PrimitiveReader::new(type_str.endianness) };
        Ok(Utf32ArrayVecReader { num_u32s_in_dtype, char_reader })
    }
}

/// _This impl is only available with the **`"arrayvec"`** feature._
#[cfg(feature = "arrayvec")]
impl<const N: usize> Deserialize for ArrayString<N> {
    type TypeReader = Utf8ArrayStringReader<N>;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, DTypeError> {
        match Utf8StringReader::try_from_dtype(dtype) {
            Some(string_reader) => Ok(Utf8ArrayStringReader { string_reader: string_reader? }),
            None => {
                let type_str = expect_scalar_dtype::<Self>(dtype)?;
                Err(DTypeError::bad_scalar::<Self>("read", &type_str))
            },
        }
    }
}

/// Writes `Vec<u32>` to `U`, permitting surrogates.
#[doc(hidden)]
pub struct Utf32WithSurrogatesWriter {
    int_writer: PrimitiveWriter<u32>,
    type_str: TypeStr,
    num_u32s: usize,
}

#[doc(hidden)]
pub struct Utf32Writer {
    int_writer: PrimitiveWriter<u32>,
    type_str: TypeStr,
    num_u32s: usize,
}

#[doc(hidden)]
pub struct Utf32StrWriter {
    int_writer: PrimitiveWriter<u32>,
    type_str: TypeStr,
    num_u32s: usize,
}

#[doc(hidden)]
pub struct Utf8StrWriter {
    type_str: TypeStr,
    num_bytes: usize,
}

#[doc(hidden)]
pub enum StrWriter {
    Utf8(Utf8StrWriter),
    Utf32(Utf32StrWriter),
}

impl TypeWrite for Utf32WithSurrogatesWriter {
    type Value = [u32];

    fn write_one<W: io::Write>(&self, mut w: W, u32s: &[u32]) -> io::Result<()> {
        if u32s.len() > self.num_u32s {
            return Err(bad_length(u32s, &self.type_str));
        }

        for &dword in u32s {
            let dword = validate_type_u_code_unit(dword)?;
            self.int_writer.write_one(&mut w, &dword)?;
        }
        for _ in u32s.len()..self.num_u32s {
            self.int_writer.write_one(&mut w, &0)?;
        }
        Ok(())
    }
}

impl TypeWrite for Utf32Writer {
    type Value = [char];

    fn write_one<W: io::Write>(&self, mut w: W, utf32: &[char]) -> io::Result<()> {
        if utf32.len() > self.num_u32s {
            return Err(bad_length(utf32, &self.type_str));
        }

        for &char in utf32 {
            self.int_writer.write_one(&mut w, &(char as u32))?;
        }
        for _ in utf32.len()..self.num_u32s {
            self.int_writer.write_one(&mut w, &0)?;
        }
        Ok(())
    }
}

impl TypeWrite for Utf32StrWriter {
    type Value = str;

    fn write_one<W: io::Write>(&self, mut w: W, str: &str) -> io::Result<()> {
        let mut str_len = 0;
        for char in str.chars() {
            str_len += 1;
            if str_len > self.num_u32s {
                return Err(invalid_data(format_args!(
                    "string has too many code units ({}) for type-string '{}'",
                    str.chars().count(), &self.type_str,
                )));
            }
            self.int_writer.write_one(&mut w, &(char as u32))?;
        }
        for _ in str_len..self.num_u32s {
            self.int_writer.write_one(&mut w, &0)?;
        }
        Ok(())
    }
}

impl TypeWrite for Utf8StrWriter {
    type Value = str;

    fn write_one<W: io::Write>(&self, mut w: W, str: &str) -> io::Result<()> {
        if str.len() > self.num_bytes {
            return Err(invalid_data(format_args!(
                "string has too many code units ({}) for type-string '{}'",
                str.chars().count(), &self.type_str,
            )));
        }
        w.write_all(str.as_bytes())?;
        if str.len() < self.num_bytes {
            w.write_all(&vec![0; self.num_bytes - str.len()])?;
        }
        Ok(())
    }
}

impl TypeWrite for StrWriter {
    type Value = str;

    fn write_one<W: io::Write>(&self, w: W, str: &str) -> io::Result<()> {
        match self {
            StrWriter::Utf8(imp) => imp.write_one(w, str),
            StrWriter::Utf32(imp) => imp.write_one(w, str),
        }
    }
}

impl Serialize for [u32] {
    type TypeWriter = Utf32WithSurrogatesWriter;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        if type_str.type_char != TypeChar::UnicodeStr {
            return Err(DTypeError::bad_scalar::<Self>("write", &type_str));
        };

        let num_u32s = size_field_as_usize(type_str)?;
        let type_str = type_str.clone();
        let int_writer = PrimitiveWriter::new(type_str.endianness);
        Ok(Utf32WithSurrogatesWriter { int_writer, type_str, num_u32s })
    }
}

impl Serialize for [char] {
    type TypeWriter = Utf32Writer;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        if type_str.type_char != TypeChar::UnicodeStr {
            return Err(DTypeError::bad_scalar::<Self>("write", &type_str));
        };

        let num_u32s = size_field_as_usize(type_str)?;
        let type_str = type_str.clone();
        let int_writer = PrimitiveWriter::new(type_str.endianness);
        Ok(Utf32Writer { int_writer, type_str, num_u32s })
    }
}

impl Serialize for str {
    type TypeWriter = StrWriter;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, DTypeError> {
        let type_str = expect_scalar_dtype::<Self>(dtype)?;
        match type_str.type_char {
            TypeChar::UnicodeStr => {
                let num_u32s = size_field_as_usize(type_str)?;
                let type_str = type_str.clone();
                let int_writer = PrimitiveWriter::new(type_str.endianness);
                Ok(StrWriter::Utf32(Utf32StrWriter { int_writer, type_str, num_u32s }))
            },
            TypeChar::ByteStr => {
                let num_bytes = size_field_as_usize(type_str)?;
                let type_str = type_str.clone();
                Ok(StrWriter::Utf8(Utf8StrWriter { type_str, num_bytes }))
            },
            _ => Err(DTypeError::bad_scalar::<Self>("write", &type_str)),
        }
    }
}

/// Validate a code unit of a `U` string.  (i.e. UTF-32 with surrogates; basically we only need
/// to check that it is a valid codepoint)
fn validate_type_u_code_unit(value: u32) -> Result<u32, io::Error> {
    if value < 0x110000 {
        Ok(value)
    } else {
        Err(invalid_data(format_args!("codepoint {:#x} is out of range", value)))
    }
}

fn size_field_as_usize(type_str: &TypeStr) -> Result<usize, DTypeError> {
    usize::try_from(type_str.size).map_err(|_| DTypeError::bad_usize(type_str.size))
}

fn truncate_trailing_nuls<T>(vec: &mut Vec<T>, mut is_null: impl FnMut(&T) -> bool) {
    let end = vec.iter().rposition(|x| !is_null(x)).map_or(0, |ind| ind + 1);
    vec.truncate(end);
}

#[cfg(feature = "arrayvec")]
fn arrayvec_truncate_trailing_nuls<T, const N: usize>(vec: &mut ArrayVec<T, N>, mut is_null: impl FnMut(&T) -> bool) {
    let end = vec.iter().rposition(|x| !is_null(x)).map_or(0, |ind| ind + 1);
    vec.truncate(end);
}

fn bad_length<T>(slice: &[T], type_str: &TypeStr) -> io::Error {
    invalid_data(format_args!("bad item length {} for type-string '{}'", slice.len(), type_str))
}

impl_serialize_by_deref!{[] Vec<u8> => [u8]}
impl_serialize_by_deref!{[] Vec<u32> => [u32]}
impl_serialize_by_deref!{[] Vec<char> => [char]}
impl_serialize_by_deref!{[] String => str}

#[cfg(feature = "arrayvec")]
mod arrayvec_serialize_impls {
    use super::*;
    use crate::serialize::AutoSerialize;

    impl_serialize_by_deref!{
        /// _This impl is only available with the **`arrayvec`** feature._
        [const N: usize] ArrayVec<u32, N> => [u32]
    }
    impl_serialize_by_deref!{
        /// _This impl is only available with the **`arrayvec`** feature._
        [const N: usize] ArrayVec<char, N> => [char]
    }
    impl_serialize_by_deref!{
        /// _This impl is only available with the **`arrayvec`** feature._
        [const N: usize] ArrayString<N> => str
    }

    /// _This impl is only available with the **`arrayvec`** feature._
    impl<const N: usize> AutoSerialize for ArrayVec<u32, N> {
        fn default_dtype() -> DType {
            let size = u64::try_from(N).unwrap();
            DType::new_scalar(TypeStr::with_auto_endianness(TypeChar::UnicodeStr, size, None))
        }
    }

    /// _This impl is only available with the **`arrayvec`** feature._
    impl<const N: usize> AutoSerialize for ArrayVec<char, N> {
        fn default_dtype() -> DType {
            let size = u64::try_from(N).unwrap();
            DType::new_scalar(TypeStr::with_auto_endianness(TypeChar::UnicodeStr, size, None))
        }
    }

    /// _This impl is only available with the **`arrayvec`** feature._
    impl<const N: usize> AutoSerialize for ArrayString<N> {
        fn default_dtype() -> DType {
            let size = u64::try_from(N).unwrap();
            DType::new_scalar(TypeStr::with_auto_endianness(TypeChar::ByteStr, size, None))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::test_helpers::*;

    fn char_vec(str: &str) -> Vec<char> {
        str.chars().collect()
    }

    #[test]
    fn bytes_any_endianness() {
        for ty in vec!["'<S3'", "'>S3'", "'|S3'"] {
            let ty = DType::parse(ty).unwrap();
            assert_eq!(writer_output::<[u8]>(&ty, &[1, 3, 5][..]), blob![1, 3, 5]);
            assert_eq!(reader_output::<Vec<u8>>(&ty, &[1, 3, 5][..]), blob![1, 3, 5]);
        }
    }

    #[test]
    fn length_zero() {
        let ts = DType::parse("'|S0'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[]), vec![]);
        assert_eq!(writer_output::<[u8]>(&ts, &[]), blob![]);

        let ts = DType::parse("'|V0'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &[]), vec![]);
        assert_eq!(writer_output::<[u8]>(&ts, &[]), blob![]);

        let ts = DType::parse("'|V0'").unwrap();
        assert_eq!(reader_output::<FixedSizeBytes<0>>(&ts, &[]), FixedSizeBytes([]));
        assert_eq!(writer_output::<FixedSizeBytes<0>>(&ts, &FixedSizeBytes([])), blob![]);

        let ts = DType::parse("'>U0'").unwrap();
        assert_eq!(reader_output::<Vec<u32>>(&ts, &[]), vec![]);
        assert_eq!(writer_output::<[u32]>(&ts, &[]), blob![]);

        let ts = DType::parse("'>U0'").unwrap();
        assert_eq!(reader_output::<Vec<char>>(&ts, &[]), vec![]);
        assert_eq!(writer_output::<[char]>(&ts, &[]), blob![]);

        let ts = DType::parse("'>U0'").unwrap();
        assert_eq!(reader_output::<String>(&ts, &[]), String::new());
        assert_eq!(writer_output::<str>(&ts, ""), blob![]);

        let ts = DType::parse("'|S0'").unwrap();
        assert_eq!(reader_output::<String>(&ts, &[]), String::new());
        assert_eq!(writer_output::<str>(&ts, ""), blob![]);
    }

    // tests for null padding and rejection of inputs that don't fit
    #[test]
    fn write_wrong_length() {
        let s_3 = DType::parse("'|S3'").unwrap();
        assert_eq!(DType::parse("'|a3'").unwrap(), s_3);  // doesn't need its own tests
        let v_3 = DType::parse("'|V3'").unwrap();
        let u_3 = DType::parse("'>U3'").unwrap();

        assert_eq!(writer_output::<[u8]>(&s_3, &[1, 3, 5]), blob![1, 3, 5]);
        assert_eq!(writer_output::<[u8]>(&v_3, &[1, 3, 5]), blob![1, 3, 5]);
        assert_eq!(writer_output::<[u32]>(&u_3, &[1, 3, 5]), blob![be(1_u32), be(3_u32), be(5_u32)]);
        assert_eq!(writer_output::<[char]>(&u_3, &char_vec("\x01\x03\x05")[..]), blob![be(1_u32), be(3_u32), be(5_u32)]);
        assert_eq!(writer_output::<str>(&u_3, "\x01\x03\x05"), blob![be(1_u32), be(3_u32), be(5_u32)]);
        assert_eq!(writer_output::<str>(&s_3, "\x01\x03\x05"), blob![1, 3, 5]);

        assert_eq!(writer_output::<[u8]>(&s_3, &[1]), blob![1, 0, 0]);
        writer_expect_write_err::<[u8]>(&v_3, &[1]);
        assert_eq!(writer_output::<[u32]>(&u_3, &[1]), blob![be(1_u32), be(0_u32), be(0_u32)]);
        assert_eq!(writer_output::<[char]>(&u_3, &char_vec("\x01")), blob![be(1_u32), be(0_u32), be(0_u32)]);
        assert_eq!(writer_output::<str>(&u_3, "\x01"), blob![be(1_u32), be(0_u32), be(0_u32)]);
        assert_eq!(writer_output::<str>(&s_3, "\x01"), blob![1, 0, 0]);

        assert_eq!(writer_output::<[u8]>(&s_3, &[]), blob![0, 0, 0]);
        writer_expect_write_err::<[u8]>(&v_3, &[]);
        assert_eq!(writer_output::<[u32]>(&u_3, &[]), blob![be(0_u32), be(0_u32), be(0_u32)]);
        assert_eq!(writer_output::<[char]>(&u_3, &[]), blob![be(0_u32), be(0_u32), be(0_u32)]);
        assert_eq!(writer_output::<str>(&u_3, ""), blob![be(0_u32), be(0_u32), be(0_u32)]);
        assert_eq!(writer_output::<str>(&s_3, ""), blob![0, 0, 0]);

        writer_expect_write_err::<[u8]>(&s_3, &[1, 3, 5, 7]);
        writer_expect_write_err::<[u8]>(&v_3, &[1, 3, 5, 7]);
        writer_expect_write_err::<[u32]>(&u_3, &[1, 3, 5, 7]);
        writer_expect_write_err::<[char]>(&u_3, &char_vec("\x01\x03\x05\x07"));
        writer_expect_write_err::<str>(&u_3, "\x01\x03\x05\x07");
        writer_expect_write_err::<str>(&s_3, "\x01\x03\x05\x07");
    }

    #[test]
    #[cfg(feature = "arrayvec")]
    fn write_wrong_length_arrayvec() {
        let s_3 = DType::parse("'|S3'").unwrap();
        let v_3 = DType::parse("'|V3'").unwrap();

        // Most arrayvec write impls defer to the slice implementation just like Vec, so they are
        // already tested by the `write_wrong_length`. However, ArrayVec<u8> has additional
        // preconditions for |V.
        assert_eq!(writer_output(&s_3, &ArrayVec::<u8, 2>::from_iter([1, 3])), blob![1, 3, 0]);
        assert_eq!(writer_output(&s_3, &ArrayVec::<u8, 2>::from_iter([1])), blob![1, 0, 0]);
        assert_eq!(writer_output(&s_3, &ArrayVec::<u8, 3>::from_iter([1, 3, 5])), blob![1, 3, 5]);
        assert_eq!(writer_output(&s_3, &ArrayVec::<u8, 3>::from_iter([1])), blob![1, 0, 0]);
        writer_expect_write_err(&s_3, &ArrayVec::<u8, 4>::from_iter([1, 3, 5, 7]));
        assert_eq!(writer_output(&s_3, &ArrayVec::<u8, 4>::from_iter([1])), blob![1, 0, 0]);

        writer_expect_err::<ArrayVec<u8, 2>>(&v_3);
        assert_eq!(writer_output(&v_3, &ArrayVec::<u8, 3>::from_iter([1, 3, 5])), blob![1, 3, 5]);
        writer_expect_write_err(&v_3, &ArrayVec::<u8, 3>::from_iter([1]));
        writer_expect_write_err(&v_3, &ArrayVec::<u8, 4>::from_iter([1, 3, 5, 7]));
        assert_eq!(writer_output(&v_3, &ArrayVec::<u8, 4>::from_iter([1, 3, 5])), blob![1, 3, 5]);
        writer_expect_write_err(&v_3, &ArrayVec::<u8, 4>::from_iter([1]));
    }

    #[test]
    fn fixed_size_bytes_restrictions() {
        let s_3 = DType::parse("'|S3'").unwrap();
        let v_3 = DType::parse("'|V3'").unwrap();

        writer_expect_err::<FixedSizeBytes<3>>(&s_3);
        writer_expect_err::<FixedSizeBytes<2>>(&v_3);
        writer_expect_err::<FixedSizeBytes<4>>(&v_3);
        assert_eq!(writer_output::<FixedSizeBytes<3>>(&v_3, &FixedSizeBytes([1, 3, 5])), blob![1, 3, 5]);
    }

    #[test]
    fn truncate_trailing_nuls() {
        let ts = DType::parse("'|S2'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &blob![1, 3]), vec![1, 3]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &blob![1, 0]), vec![1]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &blob![0, 0]), vec![]);

        let ts = DType::parse("'|V2'").unwrap();
        assert_eq!(reader_output::<Vec<u8>>(&ts, &blob![1, 3]), vec![1, 3]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &blob![1, 0]), vec![1, 0]);
        assert_eq!(reader_output::<Vec<u8>>(&ts, &blob![0, 0]), vec![0, 0]);

        let ts = DType::parse("'<U2'").unwrap();
        assert_eq!(reader_output::<Vec<u32>>(&ts, &blob![le(1u32), le(3u32)]), vec![1, 3]);
        assert_eq!(reader_output::<Vec<u32>>(&ts, &blob![le(1u32), le(0u32)]), vec![1]);
        assert_eq!(reader_output::<Vec<u32>>(&ts, &blob![le(0u32), le(0u32)]), vec![]);

        let ts = DType::parse("'<U2'").unwrap();
        assert_eq!(reader_output::<Vec<char>>(&ts, &blob![le(1u32), le(3u32)]), char_vec("\x01\x03"));
        assert_eq!(reader_output::<Vec<char>>(&ts, &blob![le(1u32), le(0u32)]), char_vec("\x01"));
        assert_eq!(reader_output::<Vec<char>>(&ts, &blob![le(0u32), le(0u32)]), char_vec(""));

        let ts = DType::parse("'<U2'").unwrap();
        assert_eq!(reader_output::<String>(&ts, &blob![le(1u32), le(3u32)]), "\x01\x03");
        assert_eq!(reader_output::<String>(&ts, &blob![le(1u32), le(0u32)]), "\x01");
        assert_eq!(reader_output::<String>(&ts, &blob![le(0u32), le(0u32)]), "");

        let ts = DType::parse("'|S2'").unwrap();
        assert_eq!(reader_output::<String>(&ts, &blob![1, 3]), "\x01\x03");
        assert_eq!(reader_output::<String>(&ts, &blob![1, 0]), "\x01");
        assert_eq!(reader_output::<String>(&ts, &blob![0, 0]), "");
    }

    #[test]
    #[cfg(feature = "arrayvec")]
    fn arrayvec_truncate_trailing_nuls() {
        let ts = DType::parse("'<U2'").unwrap();
        assert_eq!(reader_output::<ArrayVec<u32, 2>>(&ts, &blob![le(1u32), le(3u32)]), ArrayVec::from_iter(vec![1, 3]));
        assert_eq!(reader_output::<ArrayVec<u32, 2>>(&ts, &blob![le(1u32), le(0u32)]), ArrayVec::from_iter(vec![1]));
        assert_eq!(reader_output::<ArrayVec<u32, 2>>(&ts, &blob![le(0u32), le(0u32)]), ArrayVec::from_iter(vec![]));

        let ts = DType::parse("'<U2'").unwrap();
        assert_eq!(reader_output::<ArrayVec<char, 2>>(&ts, &blob![le(1u32), le(3u32)]), ArrayVec::from_iter("\x01\x03".chars()));
        assert_eq!(reader_output::<ArrayVec<char, 2>>(&ts, &blob![le(1u32), le(0u32)]), ArrayVec::from_iter("\x01".chars()));
        assert_eq!(reader_output::<ArrayVec<char, 2>>(&ts, &blob![le(0u32), le(0u32)]), ArrayVec::from_iter("".chars()));

        let ts = DType::parse("'|S2'").unwrap();
        assert_eq!(reader_output::<ArrayString<2>>(&ts, &blob![1, 3]), ArrayString::<2>::try_from("\x01\x03").unwrap());
        assert_eq!(reader_output::<ArrayString<2>>(&ts, &blob![1, 0]), ArrayString::<2>::try_from("\x01").unwrap());
        assert_eq!(reader_output::<ArrayString<2>>(&ts, &blob![0, 0]), ArrayString::<2>::try_from("").unwrap());

        let ts = DType::parse("'|S2'").unwrap();
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &blob![1, 3]), ArrayVec::<u8, 2>::from_iter(vec![1, 3]));
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &blob![1, 0]), ArrayVec::<u8, 2>::from_iter(vec![1]));
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &blob![0, 0]), ArrayVec::<u8, 2>::from_iter(vec![]));

        let ts = DType::parse("'|V2'").unwrap();
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &blob![1, 3]), ArrayVec::<u8, 2>::from_iter(vec![1, 3]));
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &blob![1, 0]), ArrayVec::<u8, 2>::from_iter(vec![1, 0]));
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &blob![0, 0]), ArrayVec::<u8, 2>::from_iter(vec![0, 0]));
    }

    #[test]
    fn preserve_interior_nuls() {
        let value: &[u8] = &[0, 1, 0, 0, 3, 5];
        let encoded: &[u8] = &blob![0, 1, 0, 0, 3, 5];
        let ts = DType::parse("'|S6'").unwrap();

        assert_eq!(reader_output::<Vec<u8>>(&ts, encoded), value.to_vec());
        assert_eq!(writer_output(&ts, value), encoded.to_vec());

        let value: &[u32] = &[0, 1, 0, 0, 3, 5];
        let encoded: &[u8] = &blob![le(0_u32), le(1_u32), le(0_u32), le(0_u32), le(3_u32), le(5_u32)];
        let ts = DType::parse("'<U6'").unwrap();

        assert_eq!(reader_output::<Vec<u32>>(&ts, encoded), value.to_vec());
        assert_eq!(writer_output(&ts, value), encoded.to_vec());

        let value: &[char] = &char_vec("\x00\x01\x00\x00\x03\x05")[..];
        let encoded: &[u8] = &blob![le(0_u32), le(1_u32), le(0_u32), le(0_u32), le(3_u32), le(5_u32)];
        let ts = DType::parse("'<U6'").unwrap();

        assert_eq!(reader_output::<Vec<char>>(&ts, encoded), value.to_vec());
        assert_eq!(writer_output(&ts, value), encoded.to_vec());

        let value: &str = "\x00\x01\x00\x00\x03\x05";
        let encoded: &[u8] = &blob![le(0_u32), le(1_u32), le(0_u32), le(0_u32), le(3_u32), le(5_u32)];
        let ts = DType::parse("'<U6'").unwrap();

        assert_eq!(reader_output::<String>(&ts, encoded), value.to_string());
        assert_eq!(writer_output(&ts, value), encoded.to_vec());

        let value: &str = "\x00\x01\x00\x00\x03\x05";
        let encoded: &[u8] = &[0, 1, 0, 0, 3, 5];
        let ts = DType::parse("'|S6'").unwrap();

        assert_eq!(reader_output::<String>(&ts, encoded), value.to_string());
        assert_eq!(writer_output(&ts, value), encoded.to_vec());
    }

    #[test]
    #[cfg(feature = "arrayvec")]
    fn arrayvec_truncated_reading() {
        let ts = DType::parse("'<U5'").unwrap();
        let data = blob![le(1_u32), le(3_u32), le(5_u32), le(7_u32), le(9_u32)];

        assert_eq!(reader_output::<ArrayVec<u32, 2>>(&ts, &data), ArrayVec::from_iter(vec![1, 3]));
        assert_eq!(reader_output::<ArrayVec<char, 2>>(&ts, &data), ArrayVec::from_iter("\x01\x03".chars()));

        let ts = DType::parse("'|S5'").unwrap();
        let data = blob![1, 3, 5, 7, 9];

        assert_eq!(reader_output::<ArrayString<2>>(&ts, &data), ArrayString::<2>::try_from("\x01\x03").unwrap());
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &data), ArrayVec::<u8, 2>::try_from(&[1, 3][..]).unwrap());
    }

    #[test]
    #[cfg(feature = "arrayvec")]
    fn arrayvec_no_array_string_from_u() {
        // We don't want ArrayString to read U because it would be nigh impossible for the caller
        // to determine when the data that we read may be truncated.
        reader_expect_ok::<ArrayString<50>>(&DType::parse("'|S6'").unwrap());
        reader_expect_err::<ArrayString<50>>(&DType::parse("'<U6'").unwrap());
        writer_expect_ok::<ArrayString<50>>(&DType::parse("'|S6'").unwrap());
        writer_expect_ok::<ArrayString<50>>(&DType::parse("'<U6'").unwrap());
    }

    #[test]
    #[cfg(feature = "arrayvec")]
    fn arrayvec_truncated_in_middle_of_utf8_char() {
        let ts = DType::parse("'|S6'").unwrap();
        let data = "abはe".as_bytes().to_vec();  // has a 3-byte char

        // ArrayString cares about UTF8
        assert_eq!(reader_output::<ArrayString<2>>(&ts, &data), ArrayString::<2>::try_from("ab").unwrap());
        assert_eq!(reader_output::<ArrayString<3>>(&ts, &data), ArrayString::<3>::try_from("ab").unwrap());
        assert_eq!(reader_output::<ArrayString<4>>(&ts, &data), ArrayString::<4>::try_from("ab").unwrap());
        assert_eq!(reader_output::<ArrayString<5>>(&ts, &data), ArrayString::<5>::try_from("abは").unwrap());

        // but ArrayVec<u8> does not
        assert_eq!(reader_output::<ArrayVec<u8, 2>>(&ts, &data), ArrayVec::<u8, 2>::try_from(&data[..2]).unwrap());
        assert_eq!(reader_output::<ArrayVec<u8, 3>>(&ts, &data), ArrayVec::<u8, 3>::try_from(&data[..3]).unwrap());
        assert_eq!(reader_output::<ArrayVec<u8, 4>>(&ts, &data), ArrayVec::<u8, 4>::try_from(&data[..4]).unwrap());
        assert_eq!(reader_output::<ArrayVec<u8, 5>>(&ts, &data), ArrayVec::<u8, 5>::try_from(&data[..5]).unwrap());
    }

    #[test]
    fn serialize_types_that_deref_to_slices() {
        let ts = DType::parse("'|S3'").unwrap();
        assert_eq!(writer_output::<Vec<u8>>(&ts, &vec![1, 3, 5]), blob![1, 3, 5]);
        assert_eq!(writer_output::<&[u8]>(&ts, &&[1, 3, 5][..]), blob![1, 3, 5]);

        let ts = DType::parse("'<U3'").unwrap();
        assert_eq!(writer_output::<Vec<u32>>(&ts, &vec![1, 3, 5]), blob![le(1_u32), le(3_u32), le(5_u32)]);
        assert_eq!(writer_output::<&[u32]>(&ts, &&[1, 3, 5][..]), blob![le(1_u32), le(3_u32), le(5_u32)]);

        let ts = DType::parse("'<U3'").unwrap();
        assert_eq!(writer_output::<Vec<char>>(&ts, &char_vec("\x01\x03\x05")), blob![le(1_u32), le(3_u32), le(5_u32)]);
        assert_eq!(writer_output::<&[char]>(&ts, &&char_vec("\x01\x03\x05")[..]), blob![le(1_u32), le(3_u32), le(5_u32)]);

        let ts = DType::parse("'<U3'").unwrap();
        assert_eq!(writer_output::<String>(&ts, &"\x01\x03\x05".to_string()), blob![le(1_u32), le(3_u32), le(5_u32)]);
        assert_eq!(writer_output::<&str>(&ts, &"\x01\x03\x05"), blob![le(1_u32), le(3_u32), le(5_u32)]);

        let ts = DType::parse("'|S3'").unwrap();
        assert_eq!(writer_output::<String>(&ts, &"\x01\x03\x05".to_string()), blob![1, 3, 5]);
        assert_eq!(writer_output::<&str>(&ts, &"\x01\x03\x05"), blob![1, 3, 5]);
    }

    const FIRST_BAD_CODEPOINT: u32 = 0x110000;
    const A_SURROGATE_CODEPOINT: u32 = 0xD805;

    #[test]
    fn reading_invalid_utf32_or_utf8() {
        let ts = DType::parse("'<U1'").unwrap();
        reader_expect_read_ok::<Vec<u32>>(&ts, &blob![le(FIRST_BAD_CODEPOINT - 1)]);
        reader_expect_read_ok::<Vec<char>>(&ts, &blob![le(FIRST_BAD_CODEPOINT - 1)]);
        reader_expect_read_ok::<String>(&ts, &blob![le(FIRST_BAD_CODEPOINT - 1)]);

        reader_expect_read_err::<Vec<u32>>(&ts, &blob![le(FIRST_BAD_CODEPOINT)]);
        reader_expect_read_err::<Vec<char>>(&ts, &blob![le(FIRST_BAD_CODEPOINT)]);
        reader_expect_read_err::<String>(&ts, &blob![le(FIRST_BAD_CODEPOINT)]);

        reader_expect_read_err::<Vec<u32>>(&ts, &blob![le(FIRST_BAD_CODEPOINT + 1)]);
        reader_expect_read_err::<Vec<char>>(&ts, &blob![le(FIRST_BAD_CODEPOINT + 1)]);
        reader_expect_read_err::<String>(&ts, &blob![le(FIRST_BAD_CODEPOINT + 1)]);

        reader_expect_read_ok::<Vec<u32>>(&ts, &blob![le(A_SURROGATE_CODEPOINT)]);
        reader_expect_read_err::<Vec<char>>(&ts, &blob![le(A_SURROGATE_CODEPOINT)]);
        reader_expect_read_err::<String>(&ts, &blob![le(A_SURROGATE_CODEPOINT)]);

        let ts = DType::parse("'|S3'").unwrap();
        reader_expect_read_err::<String>(&ts, &blob![0xFF, 0x00, 0x01]);
    }

    #[test]
    #[cfg(feature = "arrayvec")]
    fn arrayvec_reading_invalid_utf32_or_utf8() {
        let ts = DType::parse("'<U1'").unwrap();
        reader_expect_read_ok::<ArrayVec<u32, 1>>(&ts, &blob![le(FIRST_BAD_CODEPOINT - 1)]);
        reader_expect_read_ok::<ArrayVec<char, 1>>(&ts, &blob![le(FIRST_BAD_CODEPOINT - 1)]);

        reader_expect_read_err::<ArrayVec<u32, 1>>(&ts, &blob![le(FIRST_BAD_CODEPOINT)]);
        reader_expect_read_err::<ArrayVec<char, 1>>(&ts, &blob![le(FIRST_BAD_CODEPOINT)]);

        reader_expect_read_err::<ArrayVec<u32, 1>>(&ts, &blob![le(FIRST_BAD_CODEPOINT + 1)]);
        reader_expect_read_err::<ArrayVec<char, 1>>(&ts, &blob![le(FIRST_BAD_CODEPOINT + 1)]);

        reader_expect_read_ok::<ArrayVec<u32, 1>>(&ts, &blob![le(A_SURROGATE_CODEPOINT)]);
        reader_expect_read_err::<ArrayVec<char, 1>>(&ts, &blob![le(A_SURROGATE_CODEPOINT)]);

        let ts = DType::parse("'|S3'").unwrap();
        reader_expect_read_err::<ArrayString<3>>(&ts, &blob![0xFF, 0x00, 0x01]);
    }

    #[test]
    fn writing_invalid_utf32() {
        let ts = DType::parse("'<U1'").unwrap();
        writer_expect_write_ok::<[u32]>(&ts, &[FIRST_BAD_CODEPOINT - 1]);
        writer_expect_write_err::<[u32]>(&ts, &[FIRST_BAD_CODEPOINT]);
        writer_expect_write_err::<[u32]>(&ts, &[FIRST_BAD_CODEPOINT + 1]);
        writer_expect_write_ok::<[u32]>(&ts, &[A_SURROGATE_CODEPOINT]);
    }
}