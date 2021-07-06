//! Contents of `crate::npz` that require the `npz` feature, split off into
//! a separate module so that they can have a single `#[cfg(feature = "npz")]`.

use std::io;
use std::path::Path;
use std::fs::File;

use zip::result::ZipError;

use crate::read::NpyFile;
use crate::serialize::Serialize;
use crate::write::{WriterBuilder, write_options};

/// Interface for reading an NPZ file.
///
/// *This is only available with the **`"npz"`** feature.*
pub struct NpzArchive<R: io::Read + io::Seek> {
    zip: zip::ZipArchive<R>,
}

impl NpzArchive<io::BufReader<File>> {
    /// Open an `npz` archive from the filesystem.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self::new(io::BufReader::new(File::open(path)?))?)
    }
}

impl<R: io::Read + io::Seek> NpzArchive<R> {
    /// Wrap around an arbitrary stream.
    pub fn new(reader: R) -> io::Result<Self> {
        Ok(NpzArchive { zip: zip::ZipArchive::new(reader).map_err(invalid_data)? })
    }

    /// Get the names of all arrays in the NPZ file.
    pub fn array_names(&self) -> impl Iterator<Item = &str> {
        self.zip.file_names().filter_map(crate::npz::array_name_from_file_name)
    }

    /// Read the array with the given name.
    ///
    /// If it is not present, `Ok(None)` is returned.
    pub fn by_name<'a>(&'a mut self, name: &str) -> io::Result<Option<NpyFile<zip::read::ZipFile<'a>>>> {
        match self.zip.by_name(&crate::npz::file_name_from_array_name(name)) {
            Ok(file) => Ok(Some(NpyFile::new(file)?)),
            Err(ZipError::FileNotFound) => Ok(None),
            Err(ZipError::Io(e)) => Err(e),
            Err(ZipError::InvalidArchive(s)) => Err(invalid_data(s)),
            Err(ZipError::UnsupportedArchive(s)) => Err(invalid_data(s)),
        }
    }

    /// Exposes the underlying [`zip::ZipArchive`].
    pub fn zip_archive(&mut self) -> &mut zip::ZipArchive<R> {
        &mut self.zip
    }
}

fn invalid_data<S: ToString>(s: S) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.to_string())
}

/// Interface for writing an NPZ file.
///
/// *This is only available with the **`"npz"`** feature.*
pub struct NpzWriter<W: io::Write + io::Seek> {
    zip: zip::ZipWriter<W>,
}

impl NpzWriter<io::BufWriter<File>> {
    /// Create a new, empty `npz` archive on the filesystem. (will clobber an existing file)
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self::new(io::BufWriter::new(File::create(path)?)))
    }
}

impl<W: io::Write + io::Seek> NpzWriter<W> {
    /// Begin writing an NPZ file to an arbitrary writer.
    pub fn new(writer: W) -> Self {
        NpzWriter { zip: zip::ZipWriter::new(writer) }
    }

    /// Begin an entry in the NPZ for the corresponding array.
    ///
    /// The returned object implements the [`WriterBuilder`] trait.  You must import this trait
    /// and use its methods to continue configuring the object and begin writing.
    pub fn array<T: Serialize + ?Sized>(&mut self, name: &str, options: zip::write::FileOptions) -> io::Result<NpzWriterBuilder<'_, T, W>> {
        self.zip.start_file(crate::npz::file_name_from_array_name(name), options)?;
        Ok(write_options::WriteOptions::new().writer(&mut self.zip))
    }

    /// Exposes the underlying [`zip::ZipWriter`].
    pub fn zip_writer(&mut self) -> &mut zip::ZipWriter<W> {
        &mut self.zip
    }
}

/// Type returned by [`NpzWriter::array`], which implements the [`WriterBuilder`] trait.
///
/// Please use the methods of [`WriterBuilder`] to configure this object and begin writing.
/// (Note that the writer does not impl `io::Seek`, and therefore you cannot use [`WriterBuilder::begin_1d`]).
pub type NpzWriterBuilder<'w, T, W> = write_options::WithWriter<&'w mut zip::ZipWriter<W>, write_options::WriteOptions<T>>;
