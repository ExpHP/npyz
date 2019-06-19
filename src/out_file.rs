use std::io::{self,Write,BufWriter,Seek,SeekFrom};
use std::fs::File;
use std::path::Path;
use std::marker::PhantomData;

use byteorder::{WriteBytesExt, LittleEndian};

use serialize::{AutoSerialize, Serialize, TypeWrite};
use header::DType;
use npy_data::Order;

// long enough to accomodate a large integer followed by ",), }"
const FILLER: &'static [u8] = &[42; 19];

const BYTES_BEFORE_DICT: usize = 10;

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
    /// **Calling `dtype` is required.**
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Calls [`Builder::dtype`] with the default dtype for the type to be serialized.
    pub fn default_dtype(self) -> Self
    where Row: AutoSerialize,
    {
        self.dtype(Row::default_dtype())
    }
}

impl<Row: Serialize> Builder<Row> {
    // TODO: remove Seek bound via some PanicSeek newtype wrapper
    /// Begin writing an array of known shape.
    pub fn begin_nd<W: Write>(&self, w: W, shape: &[usize]) -> io::Result<NpyWriter<Row, W>> {
        NpyWriter::_begin(self, MaybeSeek::Isnt(w), Some(shape))
    }

    /// Begin writing a 1d array, of length to be inferred.
    pub fn begin_1d<W: Write + Seek>(&self, w: W) -> io::Result<NpyWriter<Row, W>> {
        NpyWriter::_begin(self, MaybeSeek::new_seek(w), None)
    }
}

/// Serialize into a file one item at a time. To serialize an iterator, use the
/// [`to_file`](fn.to_file.html) function.
pub struct NpyWriter<Row: Serialize, W: Write> {
    start_pos: u64,
    shape_info: ShapeInfo,
    num_items: usize,
    fw: MaybeSeek<W>,
    writer: <Row as Serialize>::Writer,
}

enum ShapeInfo {
    // No shape was written; we'll return to write a 1D shape on `finish()`.
    Automatic { offset_in_dict_string: u64 },
    // The complete shape has already been written.
    // Raise an error on `finish()` if the wrong number of elements is given.
    Known { expected_num_items: usize },
}

/// [`NpyWriter`] that writes an entire file.
pub type OutFile<Row> = NpyWriter<Row, BufWriter<File>>;

impl<Row: AutoSerialize> OutFile<Row> {
    /// Create a file, using the default format for the given type.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::open_with_dtype(&Row::default_dtype(), path)
    }
}

impl<Row: Serialize> OutFile<Row> {
    /// Create a file, using the provided dtype.
    pub fn open_with_dtype<P: AsRef<Path>>(dtype: &DType, path: P) -> io::Result<Self> {
        Builder::new()
            .dtype(dtype.clone())
            .begin_1d(BufWriter::new(File::create(path)?))
    }

    /// Finish writing the file and close it.  Alias for [`NpyWriter::finish`].
    ///
    /// If omitted, the file will be closed on drop automatically, ignoring any errors
    /// encountered during the process.
    pub fn close(self) -> io::Result<()> {
        self.finish()
    }
}

impl<Row: Serialize, W: Write + Seek> NpyWriter<Row, W> {
    /// Construct around an existing writer, using the default format for the given type.
    ///
    /// The header will be written immediately.
    pub fn begin(fw: W) -> io::Result<Self> where Row: AutoSerialize {
        Builder::new()
            .default_dtype()
            .begin_1d(fw)
    }

    /// Construct around an existing writer.
    ///
    /// The header will be written immediately.
    pub fn begin_with_dtype(dtype: &DType, fw: W) -> io::Result<Self> {
        Builder::new()
            .dtype(dtype.clone())
            .begin_1d(fw)
    }
}

impl<Row: Serialize, W: Write> NpyWriter<Row, W> {
    fn _begin(builder: &Builder<Row>, mut fw: MaybeSeek<W>, shape: Option<&[usize]>) -> io::Result<Self> {
        let &Builder { ref dtype, order, _marker } = builder;
        let dtype = dtype.as_ref().expect("Builder::dtype was never called!");

        let start_pos = fw.seek(SeekFrom::Current(0))?;

        if let DType::Plain { ref shape, .. } = dtype {
            assert!(shape.len() == 0, "plain non-scalar dtypes not supported");
        }
        fw.write_all(&[0x93u8])?;
        fw.write_all(b"NUMPY")?;
        fw.write_all(&[0x01u8, 0x00])?;

        let (dict_bytes, shape_info) = create_dict(dtype, order, shape);

        let writer = match Row::writer(dtype) {
            Ok(writer) => writer,
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        };

        let mut padding: Vec<u8> = vec![];
        padding.extend(&::std::iter::repeat(b' ').take(15 - ((dict_bytes.len() + BYTES_BEFORE_DICT) % 16)).collect::<Vec<_>>());
        padding.extend(&[b'\n']);

        let len = dict_bytes.len() + padding.len();
        assert! (len <= ::std::u16::MAX as usize);
        assert_eq!((len + BYTES_BEFORE_DICT) % 16, 0);

        fw.write_u16::<LittleEndian>(len as u16)?;
        fw.write_all(&dict_bytes)?;
        // Padding to 16 bytes
        fw.write_all(&padding)?;

        Ok(NpyWriter {
            start_pos: start_pos,
            shape_info: shape_info,
            num_items: 0,
            fw: fw,
            writer: writer,
        })
    }

    /// Append a single row to the file
    pub fn push(&mut self, row: &Row) -> io::Result<()> {
        self.num_items += 1;
        self.writer.write_one(&mut self.fw, row)
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
            ShapeInfo::Automatic { offset_in_dict_string } => {
                // Write the size to the header
                let shape_pos = self.start_pos + BYTES_BEFORE_DICT as u64 + offset_in_dict_string;
                let end_pos = self.fw.seek(SeekFrom::Current(0))?;

                self.fw.seek(SeekFrom::Start(shape_pos))?;
                let length = format!("{}", self.num_items);
                self.fw.write_all(length.as_bytes())?;
                self.fw.write_all(&b",), }"[..])?;
                self.fw.write_all(&::std::iter::repeat(b' ').take(FILLER.len() - length.len()).collect::<Vec<_>>())?;
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

fn create_dict(dtype: &DType, order: Order, shape: Option<&[usize]>) -> (Vec<u8>, ShapeInfo) {
    let mut header: Vec<u8> = vec![];
    header.extend(&b"{'descr': "[..]);
    header.extend(dtype.descr().as_bytes());
    header.extend(&b", 'fortran_order': "[..]);
    match order {
        Order::C => header.extend(&b"False"[..]),
        Order::Fortran => header.extend(&b"True"[..]),
    }
    header.extend(&b", 'shape': ("[..]);
    match shape {
        Some(shape) => {
            for x in shape {
                write!(header, "{}, ", x).unwrap();
            }
            header.extend(&b"), }"[..]);
            (header, ShapeInfo::Known { expected_num_items: shape.iter().product() })
        },
        None => {
            let shape_offset = header.len() as u64;
            header.extend(FILLER);
            header.extend(&b",), }"[..]);
            (header, ShapeInfo::Automatic { offset_in_dict_string: shape_offset })
        },
    }
}

impl<Row: Serialize, W: Write> Drop for NpyWriter<Row, W> {
    fn drop(&mut self) {
        let _ = self.finish_(); // Ignore the errors
    }
}


// TODO: improve the interface to avoid unnecessary clones
/// Serialize an iterator over a struct to a NPY file
///
/// A single-statement alternative to saving row by row using the [`OutFile`](struct.OutFile.html).
pub fn to_file<S, T, P>(filename: P, data: T) -> ::std::io::Result<()> where
        P: AsRef<Path>,
        S: AutoSerialize,
        T: IntoIterator<Item=S> {

    let mut of = OutFile::open(filename)?;
    for row in data {
        of.push(&row)?;
    }
    of.close()
}

use self::maybe_seek::MaybeSeek;
mod maybe_seek {
    use super::*;

    pub(crate) trait WriteSeek<W>: Write + Seek {}

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
                std::mem::transmute::<
                    Box<dyn WriteSeek<W> + '_>,
                    Box<dyn WriteSeek<W> + 'static>,
                >(Box::new(w))
            };
            MaybeSeek::Is(inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};
    use ::NpyData;

    fn bytestring_contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        haystack.windows(needle.len()).any(move |w| w == needle)
    }

    #[test]
    fn write_1d_simple() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        {
            let mut writer = NpyWriter::begin(&mut cursor)?;
            for x in vec![1.0, 3.0, 5.0] {
                writer.push(&x)?;
            }
            writer.finish()?;
        }

        let raw_buffer = cursor.into_inner();
        let reader = NpyData::<f64>::from_bytes(&raw_buffer)?;
        assert_eq!(reader.to_vec(), vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn write_1d_in_the_middle() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let prefix = b"lorem ipsum dolor sit amet.";
        let suffix = b"and they lived happily ever after.";

        // write to the cursor both before and after writing the file
        cursor.write_all(prefix)?;
        {
            let mut writer = NpyWriter::begin(&mut cursor)?;
            for x in vec![1.0, 3.0, 5.0] {
                writer.push(&x)?;
            }
            writer.finish()?;
        }
        cursor.write_all(suffix)?;

        // check that `OutFile` did not interfere with our extra writes
        let raw_buffer = cursor.into_inner();
        assert!(raw_buffer.starts_with(prefix));
        assert!(raw_buffer.ends_with(suffix));

        // check the bytes written by `OutFile`
        let written_bytes = &raw_buffer[prefix.len()..raw_buffer.len() - suffix.len()];
        let reader = NpyData::<f64>::from_bytes(&written_bytes)?;
        assert_eq!(reader.to_vec(), vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn implicit_finish() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        {
            let mut writer = NpyWriter::begin(&mut cursor)?;
            for x in vec![1.0, 3.0, 5.0, 7.0] {
                writer.push(&x)?;
            }
            // don't call finish
        }

        // check that the shape was written
        let raw_buffer = cursor.into_inner();
        println!("{:?}", raw_buffer);
        assert!(bytestring_contains(&raw_buffer, b"'shape': (4,"));

        Ok(())
    }

    #[test]
    fn write_nd_simple() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        {
            let mut writer = Builder::new().default_dtype().begin_nd(&mut cursor, &[2, 3])?;
            for x in vec![00, 01, 02, 10, 11, 12] {
                writer.push(&x)?;
            }
            writer.finish()?;
        }

        let raw_buffer = cursor.into_inner();
        let reader = NpyData::<i32>::from_bytes(&raw_buffer)?;
        assert_eq!(reader.to_vec(), vec![00, 01, 02, 10, 11, 12]);
        assert_eq!(reader.shape(), &[2, 3][..]);

        Ok(())
    }

    #[test]
    fn write_nd_wrong_len() -> io::Result<()> {
        let try_writing = |elems: &[i32]| -> io::Result<()> {
            let mut cursor = Cursor::new(vec![]);
            let mut writer = Builder::new().default_dtype().begin_nd(&mut cursor, &[2, 3])?;
            for &x in elems {
                writer.push(&x)?;
            }
            writer.finish()?;
            Ok(())
        };
        assert!(try_writing(&[00, 01, 02, 10, 11]).is_err());
        assert!(try_writing(&[00, 01, 02, 10, 11, 12]).is_ok());
        assert!(try_writing(&[00, 01, 02, 10, 11, 12, 20]).is_err());

        Ok(())
    }
}
