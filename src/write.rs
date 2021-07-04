use std::io::{self,Write,BufWriter,Seek,SeekFrom};
use std::fs::File;
use std::path::Path;
use std::marker::PhantomData;

use byteorder::{WriteBytesExt, LittleEndian};

use crate::serialize::{AutoSerialize, Serialize, TypeWrite};
use crate::header::{self, DType, VersionProps, HeaderSizeType, HeaderEncoding};
use crate::read::Order;

// Long enough to accomodate a large integer followed by ",), }".
// Used when no shape is provided.
const FILLER_FOR_UNKNOWN_SIZE: &'static [u8] = &[b'*'; 19];

/// Builder for an output `.NPY` file.
pub struct Builder<Row> {
    order: Order,
    dtype: Option<DType>,
    _marker: PhantomData<fn(Row)>, // contravariant
}

impl<Row> Builder<Row> {
    /// Construct a builder with default configuration.
    ///
    /// Data order will be initially set to C order.
    ///
    /// No dtype will be configured; the [`Builder::dtype`] method **must** be called.
    pub fn new() -> Self {
        Builder {
            order: Order::C,
            dtype: None,
            _marker: PhantomData,
        }
    }

    /// Set the data order for arrays with more than one dimension.
    ///
    /// If this is not called, `Order::C` is assumed.
    pub fn order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }

    /// Use the specified dtype.
    ///
    /// **Calling `dtype` (or [`Self::default_dtype`]) is required.**
    /// (Otherwise, the build methods will panic at runtime.)
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Calls [`Builder::dtype`] with the default dtype for the type to be serialized.
    ///
    /// **Calling this method or [`Builder::dtype`] is required.**
    /// (Otherwise, the build methods will panic at runtime.)
    pub fn default_dtype(self) -> Self
    where Row: AutoSerialize,
    {
        self.dtype(Row::default_dtype())
    }
}

impl<Row: Serialize> Builder<Row> {
    /// Begin writing an array of known shape.
    pub fn begin_nd<W: Write>(&self, w: W, shape: &[u64]) -> io::Result<NpyWriter<Row, W>> {
        NpyWriter::_begin(self, MaybeSeek::Isnt(w), Some(shape))
    }

    /// Begin writing a 1d array, of length to be inferred from the number of elements written.
    ///
    /// Notice that, in contrast to [`Self::begin_nd`], this method requires [`Seek`].  If you have
    /// a `Vec<u8>`, you can wrap it in an [`io::Cursor`] to satisfy this requirement.
    pub fn begin_1d<W: Write + Seek>(&self, w: W) -> io::Result<NpyWriter<Row, W>> {
        NpyWriter::_begin(self, MaybeSeek::new_seek(w), None)
    }
}

/// Serialize into a file one item at a time. To serialize an iterator, use the
/// [`to_file`](fn.to_file.html) function.
pub struct NpyWriter<Row: Serialize, W: Write> {
    start_pos: Option<u64>,
    shape_info: ShapeInfo,
    num_items: u64,
    fw: MaybeSeek<W>,
    writer: <Row as Serialize>::TypeWriter,
    version_props: VersionProps,
}

enum ShapeInfo {
    // No shape was written; we'll return to write a 1D shape on `finish()`.
    Automatic { offset_in_header_text: u64 },
    // The complete shape has already been written.
    // Raise an error on `finish()` if the wrong number of elements is given.
    Known { expected_num_items: u64 },
}

/// [`NpyWriter`] that writes an entire file.
#[deprecated(since = "0.5.0", note = "Doesn't carry its weight.  Use to_file_1d instead, or replicate the original behavior with Builder::new().default_dtype().begin_1d(std::io::BufWriter::new(std::fs::File::create(path)?))")]
pub type OutFile<Row> = NpyWriter<Row, BufWriter<File>>;

#[allow(deprecated)]
impl<Row: AutoSerialize> OutFile<Row> {
    /// Create a file, using the default format for the given type.
    #[deprecated(since = "0.5.0", note = "Doesn't carry its weight.  Use to_file_1d instead, or replicate the original behavior with Builder::new().default_dtype().begin_1d(std::io::BufWriter::new(std::fs::File::create(path)?))")]
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Builder::new()
            .default_dtype()
            .begin_1d(BufWriter::new(File::create(path)?))
    }
}

#[allow(deprecated)]
impl<Row: Serialize> OutFile<Row> {
    /// Finish writing the file and close it.  Alias for [`NpyWriter::finish`].
    ///
    /// If omitted, the file will be closed on drop automatically, ignoring any errors
    /// encountered during the process.
    #[deprecated(since = "0.5.0", note = "use .finish() instead")]
    pub fn close(self) -> io::Result<()> {
        self.finish()
    }
}

impl<Row: Serialize, W: Write> NpyWriter<Row, W> {
    fn _begin(builder: &Builder<Row>, mut fw: MaybeSeek<W>, shape: Option<&[u64]>) -> io::Result<Self> {
        let &Builder { ref dtype, order, _marker } = builder;
        let dtype = dtype.as_ref().expect("Builder::dtype was never called!");

        let start_pos = match fw {
            MaybeSeek::Is(ref mut fw) => Some(fw.seek(SeekFrom::Current(0))?),
            MaybeSeek::Isnt(_) => None,
        };

        if let DType::Plain { ref shape, .. } = dtype {
            assert!(shape.len() == 0, "plain non-scalar dtypes not supported");
        }
        let (dict_text, shape_info) = create_dict(dtype, order, shape);
        let (header_text, version, version_props) = determine_required_version_and_pad_header(dict_text);

        fw.write_all(&[0x93u8])?;
        fw.write_all(b"NUMPY")?;
        fw.write_all(&[version.0, version.1])?;

        assert_eq!((header_text.len() + version_props.bytes_before_text()) % 16, 0);
        match version_props.header_size_type {
            HeaderSizeType::U16 => {
                assert!(header_text.len() <= u16::MAX as usize);
                fw.write_u16::<LittleEndian>(header_text.len() as u16)?;
            },
            HeaderSizeType::U32 => {
                assert!(header_text.len() <= u32::MAX as usize);
                fw.write_u32::<LittleEndian>(header_text.len() as u32)?;
            },
        }
        fw.write_all(&header_text)?;

        let writer = match Row::writer(dtype) {
            Ok(writer) => writer,
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        };

        Ok(NpyWriter {
            start_pos,
            shape_info,
            num_items: 0,
            fw,
            writer,
            version_props,
        })
    }

    /// Append a single row to the file
    pub fn push(&mut self, row: &Row) -> io::Result<()> {
        self.num_items += 1;
        self.writer.write_one(&mut self.fw, row)
    }

    /// Write an iterator to the file
    pub fn extend(&mut self, rows: impl IntoIterator<Item=Row>) -> io::Result<()> {
        rows.into_iter().try_for_each(|row| self.push(&row))
    }

    fn finish_(&mut self) -> io::Result<()> {
        match self.shape_info {
            ShapeInfo::Known { expected_num_items } => {
                if expected_num_items != self.num_items {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, {
                        format!("shape has {} item(s), but {} item(s) were written!", expected_num_items, self.num_items)
                    }));
                }
            },
            ShapeInfo::Automatic { offset_in_header_text } => {
                // Write the size to the header
                let shape_pos = self.start_pos.unwrap() + self.version_props.bytes_before_text() as u64 + offset_in_header_text;
                let end_pos = self.fw.seek(SeekFrom::Current(0))?;

                self.fw.seek(SeekFrom::Start(shape_pos))?;
                let length = format!("{}", self.num_items);
                self.fw.write_all(length.as_bytes())?;
                self.fw.write_all(&b",), }"[..])?;
                self.fw.write_all(&::std::iter::repeat(b' ').take(FILLER_FOR_UNKNOWN_SIZE.len() - length.len()).collect::<Vec<_>>())?;
                self.fw.seek(SeekFrom::Start(end_pos))?;
            },
        }
        Ok(())
    }

    /// Finish writing the file.
    ///
    /// If no shape was provided, this will update the header to reflect the number of
    /// elements written. If a shape was provided and the number of inserted elements is
    /// incorrect, an error is returned.
    ///
    /// This is automatically called on drop, but in that case, errors are ignored.
    pub fn finish(mut self) -> io::Result<()> {
        self.finish_()
    }
}

fn create_dict(dtype: &DType, order: Order, shape: Option<&[u64]>) -> (Vec<u8>, ShapeInfo) {
    let mut header: Vec<u8> = vec![];
    header.extend(&b"{'descr': "[..]);
    header.extend(dtype.descr().as_bytes());
    header.extend(&b", 'fortran_order': "[..]);
    match order {
        Order::C => header.extend(&b"False"[..]),
        Order::Fortran => header.extend(&b"True"[..]),
    }
    header.extend(&b", 'shape': ("[..]);
    let shape_info = match shape {
        Some(shape) => {
            for x in shape {
                write!(header, "{}, ", x).unwrap();
            }
            header.extend(&b"), }"[..]);
            ShapeInfo::Known { expected_num_items: shape.iter().product() }
        },
        None => {
            let shape_offset = header.len() as u64;
            header.extend(FILLER_FOR_UNKNOWN_SIZE);
            header.extend(&b",), }"[..]);
            ShapeInfo::Automatic { offset_in_header_text: shape_offset }
        },
    };
    (header, shape_info)
}

impl<Row: Serialize, W: Write> Drop for NpyWriter<Row, W> {
    fn drop(&mut self) {
        let _ = self.finish_(); // Ignore the errors
    }
}

/// This does two things:
///
/// - Get the minimum version required to write a file, based on its header text.
/// - Pad the end of the header text so that the data begins aligned to 16 bytes.
///
/// Why must it do these together?  It turns out there's a tricky corner-case interaction for
/// header lengths close to but *just under* 65536, where the padding can push the length over
/// the 65536 threshold, causing version 2 to be used and therefore use 2 additional bytes.
/// Those additional bytes in turn could throw off the padding.
fn determine_required_version_and_pad_header(mut header_utf8: Vec<u8>) -> (Vec<u8>, (u8, u8), VersionProps) {
    use HeaderSizeType::*;
    use HeaderEncoding::*;

    // I'm almost 100% certain that, when regarding the initial length of dict_utf8,
    // there is a precise value at which the optimal version suddenly switches from 1 to 2.
    // I think it is either 65524, 65525, or 65526; just not sure which.  (the newline makes it weird)
    //
    // Unfortunately testing this edge case is not easy, so to be safe we'll give ourselves more wiggle
    // room than could possibly ever be affected by padding and/or pre-header bytes.   - ExpHP
    const SAFE_U16_CUTOFF: usize = 0x1_0000_0000 - 0x400;

    let required_props = VersionProps {
        header_size_type: if header_utf8.len() >= SAFE_U16_CUTOFF { U32 } else { U16 },
        encoding: if header_utf8.iter().any(|b| !b.is_ascii()) { Utf8 } else { Ascii },
    };

    let version = header::get_minimal_version(required_props);

    // Actual props may differ from required props.  (e.g. if it has unicode, then it needs
    // to use version 3 which will cause the size to be upgraded to U32 even if not needed)
    let actual_props = header::get_version_props(version).expect("generated internally so must be valid");

    // Now pad using the final choice of version.
    //
    // From the numpy documentation:
    //
    //    It is terminated by a newline (\n) and padded with spaces (\x20) to make the total of
    //    len(magic string) + 2 + len(length) + HEADER_LEN be evenly divisible by 64 for alignment purposes.
    const ALIGN_TO: usize = 64;

    let bytes_before_text = actual_props.bytes_before_text();
    header_utf8.extend(&::std::iter::repeat(b' ').take(ALIGN_TO - 1 - ((header_utf8.len() + bytes_before_text) % ALIGN_TO)).collect::<Vec<_>>());
    header_utf8.push(b'\n');
    assert_eq!((header_utf8.len() + bytes_before_text) % ALIGN_TO, 0);

    (header_utf8, version, actual_props)
}

#[deprecated(since = "0.5.0", note = "renamed to to_file_1d")]
/// Serialize an iterator over a struct to a NPY file.
///
/// This only saves a 1D array.  To save an ND array, **you must use the [`Builder`] API.**
pub fn to_file<S, T, P>(filename: P, data: T) -> std::io::Result<()>
where
    P: AsRef<Path>,
    S: AutoSerialize,
    T: IntoIterator<Item=S>,
{
    to_file_1d(filename, data)
}

/// Serialize an iterator over a struct to a NPY file.
///
/// This only saves a 1D array.  To save an ND array, **you must use the [`Builder`] API.**
pub fn to_file_1d<S, T, P>(filename: P, data: T) -> std::io::Result<()>
where
    P: AsRef<Path>,
    S: AutoSerialize,
    T: IntoIterator<Item=S>,
{
    #![allow(deprecated)]
    let mut of = OutFile::open(filename)?;
    for row in data {
        of.push(&row)?;
    }
    of.close()
}

// module encapsulating the unsafety of MaybeSeek
use maybe_seek::MaybeSeek;
mod maybe_seek {
    use super::*;

    pub(crate) trait WriteSeek<W>: Write + Seek + sealed::Sealed<W> {}

    mod sealed {
        use super::*;

        pub(crate) trait Sealed<W> {}
        impl<W: Write + Seek> Sealed<W> for W {}
    }

    impl<W: Write + Seek> WriteSeek<W> for W {}

    pub(crate) enum MaybeSeek<W> {
        Is(Box<dyn WriteSeek<W>>),
        Isnt(W),
    }

    impl<W: Write> Write for MaybeSeek<W> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            match self {
                MaybeSeek::Is(w) => (*w).write(buf),
                MaybeSeek::Isnt(w) => w.write(buf),
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            match self {
                MaybeSeek::Is(w) => (*w).flush(),
                MaybeSeek::Isnt(w) => w.flush(),
            }
        }
    }

    impl<W> Seek for MaybeSeek<W> {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            match self {
                MaybeSeek::Is(w) => (*w).seek(pos),
                MaybeSeek::Isnt(_) => unreachable!("(BUG!) .seek() called on MaybeSeek::Isnt!"),
            }
        }
    }

    impl<W: WriteSeek<W>> MaybeSeek<W> {
        pub fn new_seek(w: W) -> Self {
            let inner = unsafe {
                // The Self type is W, so all lifetime information contained in the unnamed
                // lifetime here is also contained in W.
                //
                // Because `dyn WriteSeek<W> + '_` is invariant in W, the compiler will
                // conservatively assume that it carries all borrows held by W; just as if
                // we *hadn't* erased the lifetime.
                //
                // See discussion here:
                //   https://users.rust-lang.org/t/a-trait-object-with-an-implied-lifetime/29340
                std::mem::transmute::<
                    Box<dyn WriteSeek<W> + '_>,
                    Box<dyn WriteSeek<W> + 'static>,
                >(Box::new(w))
            };
            MaybeSeek::Is(inner)
        }
    }
}

/// Quick API for writing a 1D array to a vector of bytes.
#[cfg(test)]
pub(crate) fn to_bytes_1d<T: AutoSerialize>(data: &[T]) -> io::Result<Vec<u8>> {
    let mut cursor = io::Cursor::new(vec![]);
    to_writer_1d(&mut cursor, data)?;
    Ok(cursor.into_inner())
}

/// Quick API for writing a 1D array to an io::Write.
#[cfg(test)]
pub(crate) fn to_writer_1d<W: io::Write + io::Seek, T: AutoSerialize>(writer: W, data: &[T]) -> io::Result<()> {
    // we might change this later and/or remove the Seek bound from the current function, but for now this will do
    to_writer_1d_with_seeking(writer, data)
}

/// Quick API for writing an n-d array to an io::Write.
#[cfg(test)]
pub(crate) fn to_writer_nd<W: io::Write + io::Seek, T: AutoSerialize>(writer: W, data: &[T], shape: &[u64]) -> io::Result<()> {
    let mut writer = Builder::new().default_dtype().begin_nd(writer, shape)?;
    writer.extend(data)?;
    writer.finish()
}

/// Quick API for writing a 1D array to an io::Write in a manner which makes use of io::Seek.
///
/// (tests will use this instead of 'to_writer_1d' if their purpose is to test the correctness of seek behavior,
/// so that changing 'to_writer_1d' to be Seek-less won't affect these tests)
#[cfg(test)]
pub(crate) fn to_writer_1d_with_seeking<W: io::Write + io::Seek, T: AutoSerialize>(writer: W, data: &[T]) -> io::Result<()> {
    let mut writer = Builder::new().default_dtype().begin_1d(writer)?;
    writer.extend(data)?;
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};
    use crate::NpyReader;

    fn bytestring_contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        haystack.windows(needle.len()).any(move |w| w == needle)
    }

    #[test]
    fn write_1d_simple() -> io::Result<()> {
        let raw_buffer = to_bytes_1d(&[1.0, 3.0, 5.0])?;

        let reader = NpyReader::<f64, _>::new(&raw_buffer[..])?;
        assert_eq!(reader.into_vec()?, vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn write_1d_in_the_middle() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let prefix = b"lorem ipsum dolor sit amet.";
        let suffix = b"and they lived happily ever after.";

        // write to the cursor both before and after writing the file
        cursor.write_all(prefix)?;
        to_writer_1d_with_seeking(&mut cursor, &[1.0, 3.0, 5.0])?;
        cursor.write_all(suffix)?;

        // check that the seeking did not interfere with our extra writes
        let raw_buffer = cursor.into_inner();
        assert!(raw_buffer.starts_with(prefix));
        assert!(raw_buffer.ends_with(suffix));

        // check the bytes written by `OutFile`
        let written_bytes = &raw_buffer[prefix.len()..raw_buffer.len() - suffix.len()];
        let reader = NpyReader::<f64, _>::new(&written_bytes[..])?;
        assert_eq!(reader.into_vec()?, vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn implicit_finish() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let mut writer = Builder::new().default_dtype().begin_1d(&mut cursor)?;
        writer.extend(vec![1.0, 3.0, 5.0, 7.0])?;
        // don't call finish
        drop(writer);

        // check that the shape was written
        let raw_buffer = cursor.into_inner();
        println!("{:?}", raw_buffer);
        assert!(bytestring_contains(&raw_buffer, b"'shape': (4,"));

        Ok(())
    }

    #[test]
    fn write_nd_simple() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        to_writer_nd(&mut cursor, &[00, 01, 02, 10, 11, 12], &[2, 3])?;

        let raw_buffer = cursor.into_inner();
        let reader = NpyReader::<i32, _>::new(&raw_buffer[..])?;
        assert_eq!(reader.shape(), &[2, 3][..]);
        assert_eq!(reader.into_vec()?, vec![00, 01, 02, 10, 11, 12]);

        Ok(())
    }

    #[test]
    fn write_nd_wrong_len() -> io::Result<()> {
        let try_writing = |elems: &[i32]| -> io::Result<()> {
            let mut cursor = Cursor::new(vec![]);
            let mut writer = Builder::new().default_dtype().begin_nd(&mut cursor, &[2, 3])?;
            writer.extend(elems)?;
            writer.finish()?;
            Ok(())
        };
        assert!(try_writing(&[00, 01, 02, 10, 11]).is_err());
        assert!(try_writing(&[00, 01, 02, 10, 11, 12]).is_ok());
        assert!(try_writing(&[00, 01, 02, 10, 11, 12, 20]).is_err());

        Ok(())
    }
}
