use std::collections::HashMap;
use std::io;

use crate::header::{Value, DType, read_header, convert_value_to_shape};
use crate::serialize::{Deserialize, TypeRead, DTypeError};

/// Object for reading an `npy` file.
///
/// This type represents a partially read `npy` file, where the header has been parsed
/// and we are ready to begin parsing data.
/// ```
/// # fn main() -> std::io::Result<()> {
/// use std::fs::File;
/// use std::io;
///
/// let file = io::BufReader::new(File::open("./test-data/c-order.npy")?);
/// let npy = npyz::NpyFile::new(file)?;
///
/// // Helper methods for inspecting the layout of the data.
/// assert_eq!(npy.shape(), &[2, 3, 4]);
/// assert_eq!(npy.strides(), &[12, 4, 1]);
/// assert_eq!(npy.order(), npyz::Order::C);
///
/// // Get the data!
/// let data: Vec<i64> = npy.into_vec()?;
/// assert_eq!(data.len(), 24);
/// # Ok(()) }
/// ```
///
/// # Working with large files
///
/// For large files, it may be undesirable to read all of the data into a Vec.
/// The [`NpyFile::data`] method allows you to iterate over the data instead.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use std::fs::File;
/// # use std::io;
/// #
/// # let file = io::BufReader::new(File::open("./test-data/c-order.npy")?);
/// let npy = npyz::NpyFile::new(file)?;
///
/// let mut sum = 0;
/// for x in npy.data::<i64>()? {
///     sum += x?;  // items are Result
/// }
/// assert_eq!(sum, 84);
/// # Ok(()) }
/// ```
///
/// # Related types
///
/// Is a read adaptor too heavy for you?  [`NpyFile`] is ultimately just a
/// [`NpyHeader`] paired with its input stream, so you may consider parsing
/// [`NpyHeader`] instead if you need something easier to clone or send.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use std::fs::File;
/// # use std::io;
/// #
/// let mut file = io::BufReader::new(File::open("./test-data/c-order.npy")?);
/// let header = npyz::NpyHeader::from_reader(&mut file)?;
/// assert_eq!(header.shape(), &[2, 3, 4]);
///
/// // now you can store `header` somewhere
/// // ...
/// // and later in your program you can construct an NpyFile to read the data
///
/// let npy = npyz::NpyFile::with_header(header, file);
/// let data: Vec<i64> = npy.into_vec()?;
/// assert_eq!(data.len(), 24);
/// # Ok(()) }
/// ```
///
/// # Migrating from `npy-rs 0.4.0`
///
/// [`NpyData`] is still provided, but it is deprecated in favor of [`NpyFile`].
/// At construction, since `&[u8]` impls `Read`, you can still use them as input.
///
/// ```text
/// was:
///     npyz::NpyData::<i64>::from_bytes(&bytes).to_vec()
/// now:
///     npyz::NpyFile::new(&bytes[..]).into_vec::<i64>()
/// ```
///
/// If you were using the iterator API of `NpyData`, this is now on [`NpyReader`], which
/// requires us to call [`NpyFile::data`]:
///
/// ```text
/// was:
///     let iter = npyz::NpyData::<i64>::new(&bytes)?;
/// now:
///     let iter = npyz::NpyFile::new(&bytes[..]).data::<i64>.map_err(invalid_data)?;
/// ```
///
/// where the following function has been used to paper over the fact that [`NpyFile::data`]
/// has a different Error type:
/// ```rust
/// # #[allow(unused)]
/// fn invalid_data<S: ToString>(err: S) -> std::io::Error {
///     std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string())
/// }
/// ```
///
/// [`NpyData::is_empty`] is gone due to possible ambiguity between [`NpyReader::len`] and [`NpyReader::total_len`].
/// Use the one that is appropriate for what you are doing.
///
/// If you were using [`NpyData::get`]... well, honestly, you should first consider whether
/// you could just iterate over the reader instead.  But if you were using [`NpyData::get`]
/// because you *genuinely need* random access, then there is [`NpyReader::read_at`].
///
/// ```text
/// was:
///     // note:  0 <= i < arr.len()
///     arr.get(i)
/// now:
///     // note:  0 <= i < reader.total_len()
///     reader.read_at(i)?
/// ```
#[derive(Clone)]
pub struct NpyFile<R: io::Read> {
    header: NpyHeader,
    reader: R,
}

/// Represents the parsed header portion of an `npy` file.
///
/// The header contains all of the information necessary to interpret the raw
/// data stream.  It provides the datatype, dimensions, and axis ordering of
/// the data.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use std::fs::File;
/// # use std::io;
/// #
/// let mut file = io::BufReader::new(File::open("./test-data/c-order.npy")?);
/// let header = npyz::NpyHeader::from_reader(&mut file)?;
/// assert_eq!(header.shape(), &[2, 3, 4]);
///
/// // now you can store `header` somewhere
/// // ...
/// // and later in your program you can construct an NpyFile to read the data
///
/// let npy = npyz::NpyFile::with_header(header, file);
/// let data: Vec<i64> = npy.into_vec()?;
/// assert_eq!(data.len(), 24);
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct NpyHeader {
    dtype: DType,
    shape: Vec<u64>,
    /// Strides of each axis, pre-computed from the shape/order.
    strides: Vec<u64>,
    order: Order,
    /// Total number of elements, pre-computed from the shape.
    n_records: u64,
    /// Item size in bytes.
    item_size: usize,
}

impl NpyHeader {
    /// Parse a header from the reader for an NPY file.
    ///
    /// The reader must initially be at the beginning of an NPY file. After this
    /// function returns `Ok(_)`, the reader will have been advanced to the
    /// beginning of the raw data bytes.
    pub fn from_reader(r: impl io::Read) -> io::Result<NpyHeader> {
        NpyHeader::read_and_interpret(r)
    }
}

/// Iterator returned by [`NpyFile::data`] which reads elements of type T from the
/// data portion of an NPY file.
///
/// This type is an iterator of `Result<T>`, with some additional methods for random access
/// when the underlying reader is seekable.
pub struct NpyReader<T: Deserialize, R: io::Read> {
    header: NpyHeader,
    type_reader: <T as Deserialize>::TypeReader,
    // stateful parts, put together like this to remind you to always update them in sync
    reader_and_current_index: (R, u64),
}

/// Legacy type for reading `npy` files.
///
/// > This type provides the same API for reading from `npy-rs 0.4.0`, to help with migration.
/// > It will later be removed in favor of [`NpyFile`].
///
/// The data is internally stored
/// as a byte array, and deserialized only on-demand to minimize unnecessary allocations.
/// The whole contents of the file can be deserialized by the [`NpyData::to_vec`] method.
#[deprecated(since = "0.5.0", note = "use NpyReader instead")]
pub struct NpyData<'a, T: Deserialize> {
    inner: NpyReader<T, &'a [u8]>,
}

/// Order of axes in a file.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Order {
    /// Indices are arranged from slowest to fastest, so that the last dimension has a stride of 1.
    C,
    /// Indices are arranged from fastest to slowest, so that the first dimension has a stride of 1.
    Fortran,
}

impl Order {
    pub(crate) fn from_fortran_order(fortran_order: bool) -> Order {
        if fortran_order { Order::Fortran } else { Order::C }
    }
}

impl<R: io::Read> NpyFile<R> {
    /// Read the header of an `npy` file and construct an `NpyFile` for reading the data.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let header = NpyHeader::read_and_interpret(&mut reader)?;
        Ok(NpyFile { header, reader })
    }

    /// Construct from a previously parsed header and a reader for the raw data bytes.
    pub fn with_header(header: NpyHeader, data_reader: R) -> Self {
        NpyFile { header, reader: data_reader }
    }

    /// Access the underlying [`NpyHeader`] object.
    pub fn header(&self) -> &NpyHeader {
        &self.header
    }
}

// Provided for backwards compatibility.
impl<R: io::Read> std::ops::Deref for NpyFile<R> {
    type Target = NpyHeader;

    fn deref(&self) -> &NpyHeader {
        &self.header
    }
}

impl NpyHeader {
    /// Get the dtype as written in the file.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape as written in the file.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Get strides for each of the dimensions.
    ///
    /// This is the amount by which the item index changes as you move along each dimension.
    /// It is a function of both [`Self::order`] and [`Self::shape`],
    /// provided for your convenience.
    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    /// Get whether the data is in C order or fortran order.
    pub fn order(&self) -> Order {
        self.order
    }

    /// Get the total number of elements in the file. (This is the product of [`Self::shape`])
    pub fn len(&self) -> u64 {
        self.n_records
    }
}

impl<R: io::Read> NpyFile<R> {
    /// Read all elements into a flat `Vec`, in the order they are stored as.
    ///
    /// This is a convenience wrapper around [`Self::data`] and [`Iterator::collect`].
    pub fn into_vec<T: Deserialize>(self) -> io::Result<Vec<T>> {
        match self.data() {
            Ok(r) => r.collect(),
            Err(e) => Err(invalid_data(e)),
        }
    }

    /// Produce an [`NpyReader`] to begin reading elements, if `T` can be deserialized from the file's dtype.
    ///
    /// The returned type implements [`Iterator`]`<Item=io::Result<T>>`, and provides additional methods
    /// for random access when `R: Seek`.  See [`NpyReader`] for more details.
    pub fn data<T: Deserialize>(self) -> Result<NpyReader<T, R>, DTypeError> {
        let NpyFile { reader, header } = self;
        let type_reader = T::reader(&header.dtype)?;
        Ok(NpyReader { type_reader, header, reader_and_current_index: (reader, 0) })
    }

    /// Produce an [`NpyReader`] to begin reading elements, if `T` can be deserialized from the file's dtype.
    ///
    /// This fallible form of the function returns `self` on error, so that you can try again with a different `T`.
    pub fn try_data<T: Deserialize>(self) -> Result<NpyReader<T, R>, Self> {
        let type_reader = match T::reader(&self.header.dtype) {
            Ok(r) => r,
            Err(_) => return Err(self),
        };
        let NpyFile { reader, header } = self;
        Ok(NpyReader { type_reader, header, reader_and_current_index: (reader, 0) })
    }
}

impl NpyHeader {
    fn read_and_interpret(mut r: impl io::Read) -> io::Result<NpyHeader> {
        let header = read_header(&mut r)?;

        let dict = match header {
            Value::Dict(dict) => dict
                .into_iter()
                .map(|(k, v)| Ok((k.as_string().ok_or(invalid_data("key is not string"))?.to_owned(), v)))
                .collect::<io::Result<HashMap<String, Value>>>()?,
            _ => return Err(invalid_data("expected a python dict literal")),
        };

        let expect_key = |key: &str| {
            dict.get(key).ok_or_else(|| invalid_data(format_args!("dict is missing key '{}'", key)))
        };

        let order = match expect_key("fortran_order")? {
            &Value::Boolean(b) => Order::from_fortran_order(b),
            _ => return Err(invalid_data(format_args!("'fortran_order' value is not a bool"))),
        };

        let shape = convert_value_to_shape(expect_key("shape")?)?;

        let descr: &Value = expect_key("descr")?;
        let dtype = DType::from_descr(descr)?;

        Self::from_parts(dtype, shape, order)
    }

    fn from_parts(dtype: DType, shape: Vec<u64>, order: Order) -> io::Result<NpyHeader> {
        let n_records = shape.iter().product();
        let item_size = dtype.num_bytes().ok_or_else(|| {
            invalid_data(format_args!("dtype is larger than usize!"))
        })?;
        let strides = strides(order, &shape);
        Ok(NpyHeader { dtype, shape, strides, order, n_records, item_size })
    }
}

impl<T: Deserialize, R: io::Read> NpyReader<T, R> {
    #[inline(always)]
    fn reader(&self) -> &R {
        &self.reader_and_current_index.0
    }

    /// Get the dtype as written in the file.
    pub fn dtype(&self) -> DType {
        self.header.dtype.clone()
    }

    /// Get the shape as written in the file.
    pub fn shape(&self) -> &[u64] {
        &self.header.shape
    }

    /// Returns the total number of records, including those that have already been read.
    /// (This is the product of [`NpyFile::shape`])
    pub fn total_len(&self) -> u64 {
        self.header.n_records
    }

    /// Get the remaining number of records that lie after the read cursor.
    pub fn len(&self) -> u64 {
        self.header.n_records - self.reader_and_current_index.1
    }
}

/// # Random access methods
impl<R: io::Read, T: Deserialize> NpyReader<T, R> where R: io::Seek {
    /// Move the read cursor to the item at the given index.
    ///
    /// Be aware that this will affect [`Self::len`], which is always computed
    /// from the current position.
    /// Seeking to [`Self::total_len`] is well defined.
    ///
    /// # Panics
    ///
    /// Panics if the index is greater than [`Self::total_len`].
    pub fn seek_to(&mut self, index: u64) -> io::Result<()> {
        let len = self.total_len();
        assert!(index <= len, "index out of bounds for seeking (the index is {} but the len is {})", index, len);

        let (reader, current_index) = &mut self.reader_and_current_index;
        let delta = index as i64 - *current_index as i64;
        if delta != 0 {
            reader.seek(io::SeekFrom::Current(delta * self.header.item_size as i64))?;
            *current_index = index;
        }
        Ok(())
    }

    /// Read a single item at the given position, leaving the cursor at the position after it.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds. (`>=` to [`Self::total_len`]).
    pub fn read_at(&mut self, index: u64) -> io::Result<T> {
        let len = self.total_len();
        assert!(index < len, "index out of bounds for reading (the index is {} but the len is {})", index, len);

        self.seek_to(index)?;
        self.next().unwrap()
    }
}

#[allow(deprecated)]
impl<'a, T: Deserialize> NpyData<'a, T> {
    /// Deserialize a NPY file represented as bytes
    pub fn from_bytes(bytes: &'a [u8]) -> io::Result<NpyData<'a, T>> {
        let inner = NpyFile::new(bytes)?.data().map_err(invalid_data)?;

        assert_eq!(inner.header.item_size as u64 * inner.header.n_records, inner.reader().len() as u64);
        Ok(NpyData { inner })
    }

    #[inline(always)] // this should optimize into just a copy of a pointer-sized field
    fn get_data_slice(&self) -> &'a [u8] {
        self.inner.reader()
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        // we can safely assume that the len is <= usize::MAX since the entire file has already
        // been mapped into the address space (even if not occupying physical memory)
        self.inner.total_len() as usize
    }

    /// Returns whether there are zero records in this NpyData structure.
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Gets a single data-record with the specified flat index.
    ///
    /// Returns None if the index is out of bounds.
    ///
    /// # Panics
    ///
    /// Panics if the bytes stored for the element are invalid for the dtype.
    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.len() {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Gets a single data-record with the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the bytes stored for the element are invalid for the dtype,
    /// or if the index is out of bounds.
    pub fn get_unchecked(&self, i: usize) -> T {
        let item_bytes = &self.get_data_slice()[i * self.inner.header.item_size..];
        self.inner.type_reader.read_one(item_bytes).unwrap()
    }

    /// Construct a vector with the deserialized contents of the whole file.
    ///
    /// The output is a flat vector with the elements in the same order that they are in the file.
    /// [`NpyData`] is deprecated and does not provide any tools for inspecting the shape and
    /// layout of the data, so if you want to correctly read multi-dimensional arrays you should
    /// switch to [`NpyFile`].
    pub fn to_vec(&self) -> Vec<T> {
        let &(mut reader) = self.inner.reader();
        (0..self.len()).map(|_| self.inner.type_reader.read_one(&mut reader).unwrap()).collect()
    }
}

fn strides(order: Order, shape: &[u64]) -> Vec<u64> {
    match order {
        Order::C => {
            let mut strides = prefix_products(shape.iter().rev().copied()).collect::<Vec<_>>();
            strides.reverse();
            strides
        },
        Order::Fortran => prefix_products(shape.iter().copied()).collect(),
    }
}

fn prefix_products<I: IntoIterator<Item=u64>>(iter: I) -> impl Iterator<Item=u64> {
    iter.into_iter().scan(1, |acc, x| { let old = *acc; *acc *= x; Some(old) })
}

fn invalid_data<S: ToString>(s: S) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.to_string())
}

impl<R, T> Iterator for NpyReader<T, R> where T: Deserialize, R: io::Read {
    type Item = io::Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let (reader, current_index) = &mut self.reader_and_current_index;
        if *current_index < self.header.n_records {
            *current_index += 1;
            return Some(self.type_reader.read_one(reader));
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let u64_len = self.len();
        if u64_len > usize::MAX as u64 {
            (usize::MAX, None)
        } else {
            (u64_len as usize, Some(u64_len as usize))
        }
    }
}

/// A result of NPY file deserialization.
///
/// It is an iterator to offer a lazy interface in case the data don't fit into memory.
#[deprecated(since = "0.5.0", note = "NpyData is being replaced with NpyReader.")]
pub struct IntoIter<'a, T: 'a + Deserialize> {
    #[allow(deprecated)]
    data: NpyData<'a, T>,
    i: usize,
}

#[allow(deprecated)]
impl<'a, T> IntoIter<'a, T> where T: Deserialize {
    fn new(data: NpyData<'a, T>) -> Self {
        IntoIter { data, i: 0 }
    }
}

#[allow(deprecated)]
impl<'a, T: 'a> IntoIterator for NpyData<'a, T> where T: Deserialize {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

#[allow(deprecated)]
impl<'a, T> Iterator for IntoIter<'a, T> where T: Deserialize {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        self.data.get(self.i - 1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.data.len() - self.i, Some(self.data.len() - self.i))
    }
}

#[allow(deprecated)]
impl<'a, T> ExactSizeIterator for IntoIter<'a, T> where T: Deserialize {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::to_bytes_1d;

    #[test]
    fn test_strides() {
        assert_eq!(strides(Order::C, &[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(Order::C, &[]), vec![]);
        assert_eq!(strides(Order::Fortran, &[2, 3, 4]), vec![1, 2, 6]);
        assert_eq!(strides(Order::Fortran, &[]), vec![]);
    }

    #[test]
    fn test_methods_after_partial_iteration() {
        let bytes = to_bytes_1d(&[100, 101, 102, 103, 104, 105, 106]).unwrap();
        let mut reader = NpyFile::new(&bytes[..]).unwrap().data().unwrap();

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 7);

        assert!(matches!(reader.next(), Some(Ok(100))));
        assert!(matches!(reader.next(), Some(Ok(101))));

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 5);
    }

    #[test]
    fn test_next_after_finished_iteration() {
        let bytes = to_bytes_1d(&[100, 101, 102, 103, 104, 105, 106]).unwrap();
        let mut reader = NpyFile::new(&bytes[..]).unwrap().data::<i32>().unwrap();

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 7);

        assert_eq!(reader.by_ref().count(), 7);  // run iterator to completion

        assert!(reader.next().is_none());
        assert!(reader.next().is_none());

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 0);  // make sure this didn't underflow...
    }

    #[test]
    fn test_methods_after_seek() {
        let bytes = to_bytes_1d(&[100, 101, 102, 103, 104, 105, 106]).unwrap();
        let mut reader = NpyFile::new(io::Cursor::new(&bytes[..])).unwrap().data().unwrap();

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 7);

        assert!(matches!(reader.next(), Some(Ok(100))));
        assert!(matches!(reader.next(), Some(Ok(101))));

        reader.seek_to(4).unwrap();

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 3);
        assert!(matches!(reader.next(), Some(Ok(104))));

        assert_eq!(reader.read_at(2).unwrap(), 102);
        assert_eq!(reader.len(), 4);
    }

    fn check_seek_panic_boundary(items: &[i32], index: u64) {
        let bytes = to_bytes_1d(items).unwrap();
        let mut reader = NpyFile::new(io::Cursor::new(&bytes[..])).unwrap().data::<i32>().unwrap();
        let _ = reader.seek_to(index);
    }

    fn check_read_panic_boundary(items: &[i32], index: u64) {
        let bytes = to_bytes_1d(items).unwrap();
        let mut reader = NpyFile::new(io::Cursor::new(&bytes[..])).unwrap().data::<i32>().unwrap();
        let _ = reader.read_at(index);
    }

    #[test]
    fn test_seek_boundary_ok() { check_seek_panic_boundary(&[1, 2, 3], 3) }
    #[test]
    #[should_panic]
    fn test_seek_boundary_ng() { check_seek_panic_boundary(&[1, 2, 3], 4) }

    #[test]
    fn test_read_boundary_ok() { check_read_panic_boundary(&[1, 2, 3], 2) }
    #[test]
    #[should_panic]
    fn test_read_boundary_ng() { check_read_panic_boundary(&[1, 2, 3], 3) }

    #[test]
    fn test_reusing_header() {
        let bytes = to_bytes_1d(&[100, 101, 102, 103, 104, 105, 106]).unwrap();
        let mut reader = io::Cursor::new(&bytes[..]);

        let header = NpyHeader::from_reader(&mut reader).unwrap();
        let npy_1 = NpyFile::with_header(header.clone(), reader.clone());
        let npy_2 = NpyFile::with_header(header.clone(), reader.clone());

        assert_eq!(
            npy_1.into_vec::<i32>().unwrap(),
            npy_2.into_vec::<i32>().unwrap(),
        );
    }
}
