
use std::io::{self,Write,BufWriter,Seek,SeekFrom};
use std::fs::File;
use std::path::Path;
use std::marker::PhantomData;

use byteorder::{WriteBytesExt, LittleEndian};

use serializable::Serializable;
use header::DType;

const FILLER: &'static [u8] = &[42; 19];

/// Serialize into a file one row at a time. To serialize an iterator, use the
/// [`to_file`](fn.to_file.html) function.
pub struct NpyWriter<Row: Serializable, W: Write + Seek> {
    shape_pos: u64,
    len: usize,
    fw: W,
    _t: PhantomData<Row>
}

/// [`NpyWriter`] that writes an entire file.
pub type OutFile<Row> = NpyWriter<Row, BufWriter<File>>;

impl<Row: Serializable> OutFile<Row> {
    /// Open a file
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::begin(BufWriter::new(File::create(path)?))
    }

    /// Finish writing the file and close it.  Alias for [`NpyWriter::finish`].
    ///
    /// If omitted, the file will be closed on drop automatically, ignoring any errors
    /// encountered during the process.
    pub fn close(self) -> io::Result<()> {
        self.finish()
    }
}

impl<Row: Serializable, W: Write + Seek> NpyWriter<Row, W> {
    /// Construct around an existing writer.
    ///
    /// The header will be written immediately.
    pub fn begin(mut fw: W) -> io::Result<Self> {
        let start_pos = fw.seek(SeekFrom::Current(0))?;

        let dtype = Row::dtype();
        if let &DType::Plain { ref shape, .. } = &dtype {
            assert!(shape.len() == 0, "plain non-scalar dtypes not supported");
        }
        fw.write_all(&[0x93u8])?;
        fw.write_all(b"NUMPY")?;
        fw.write_all(&[0x01u8, 0x00])?;

        let (header, shape_offset) = create_header(&dtype);
        let shape_pos = start_pos + shape_offset as u64;

        let mut padding: Vec<u8> = vec![];
        padding.extend(&::std::iter::repeat(b' ').take(15 - ((header.len() + 10) % 16)).collect::<Vec<_>>());
        padding.extend(&[b'\n']);

        let len = header.len() + padding.len();
        assert! (len <= ::std::u16::MAX as usize);
        assert_eq!((len + 10) % 16, 0);

        fw.write_u16::<LittleEndian>(len as u16)?;
        fw.write_all(&header)?;
        // Padding to 8 bytes
        fw.write_all(&padding)?;

        Ok(NpyWriter {
            shape_pos: shape_pos,
            len: 0,
            fw: fw,
            _t: PhantomData,
        })
    }

    /// Append a single row to the file
    pub fn push(&mut self, row: &Row) -> io::Result<()> {
        self.len += 1;
        row.write(&mut self.fw)
    }

    fn finish_(&mut self) -> io::Result<()> {
        // Write the size to the header
        let end_pos = self.fw.seek(SeekFrom::Current(0))?;
        self.fw.seek(SeekFrom::Start(self.shape_pos))?;
        let length = format!("{}", self.len);
        self.fw.write_all(length.as_bytes())?;
        self.fw.write_all(&b",), }"[..])?;
        self.fw.write_all(&::std::iter::repeat(b' ').take(FILLER.len() - length.len()).collect::<Vec<_>>())?;
        self.fw.seek(SeekFrom::Start(end_pos))?;
        Ok(())
    }

    /// Finish writing the file by finalizing the header.
    ///
    /// This is automatically called on drop, but in that case, errors are ignored.
    pub fn finish(mut self) -> io::Result<()> {
        self.finish_()
    }
}

fn create_header(dtype: &DType) -> (Vec<u8>, usize) {
    let mut header: Vec<u8> = vec![];
    header.extend(&b"{'descr': "[..]);
    header.extend(dtype.descr().as_bytes());
    header.extend(&b", 'fortran_order': False, 'shape': ("[..]);
    let shape_pos = header.len() + 10;
    header.extend(FILLER);
    header.extend(&b",), }"[..]);
    (header, shape_pos)
}

impl<Row: Serializable, W: Write + Seek> Drop for NpyWriter<Row, W> {
    fn drop(&mut self) {
        let _ = self.finish_(); // Ignore the errors
    }
}


// TODO: improve the interface to avoid unnecessary clones
/// Serialize an iterator over a struct to a NPY file
///
/// A single-statement alternative to saving row by row using the [`OutFile`](struct.OutFile.html).
pub fn to_file<'a, S, T, P>(filename: P, data: T) -> ::std::io::Result<()> where
        P: AsRef<Path>,
        S: Serializable + 'a,
        T: IntoIterator<Item=S> {

    let mut of = OutFile::open(filename)?;
    for row in data {
        of.push(&row)?;
    }
    of.close()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};
    use ::NpyData;

    #[test]
    fn write_simple() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let mut writer = NpyWriter::begin(&mut cursor)?;
        for x in vec![1.0, 3.0, 5.0] {
            writer.push(&x)?;
        }
        writer.finish()?;

        let raw_buffer = cursor.into_inner();
        let reader = NpyData::<f64>::from_bytes(&raw_buffer)?;
        assert_eq!(reader.to_vec(), vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn write_in_the_middle() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let prefix = b"lorem ipsum dolor sit amet.";
        let suffix = b"and they lived happily ever after.";

        // write to the cursor both before and after writing the file
        cursor.write_all(prefix)?;
        let mut writer = NpyWriter::begin(&mut cursor)?;
        for x in vec![1.0, 3.0, 5.0] {
            writer.push(&x)?;
        }
        writer.finish()?;
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
}
