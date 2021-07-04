//! Contents of `crate::npz` that require the `npz` feature, split off into
//! a separate module so that they can have a single `#[cfg(feature = "npz")]`.

use std::io;
use std::path::Path;
use std::fs::File;

use zip::result::ZipError;

use crate::read::NpyReader;
use crate::serialize::Deserialize;


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
    pub fn by_name<'a, T: Deserialize>(&'a mut self, name: &str) -> io::Result<Option<NpyReader<T, zip::read::ZipFile<'a>>>> {
        match self.zip.by_name(&crate::npz::file_name_from_array_name(name)) {
            Ok(file) => Ok(Some(NpyReader::new(file)?)),
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
