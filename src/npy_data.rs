use std::io;
use header::{Value, DType, read_header, convert_value_to_shape};
use serialize::{Deserialize, TypeRead};

/// Object for reading an `npy` file.
///
/// This type is an iterator, allowing you to lazily read one element at a time (even if
/// the file is too large to fit in memory).
/// ```
/// # extern crate npy;
/// # fn main() -> std::io::Result<()> {
/// use std::fs::File;
/// use std::io;
///
/// use npy::NpyReader;
///
/// let file = io::BufReader::new(File::open("./tests/c-order.npy")?);
/// let npy = npy::NpyReader::new(file)?;
///
/// // Helper methods for inspecting the layout of the data.
/// assert_eq!(npy.shape(), &[2, 3, 4]);
/// assert_eq!(npy.strides(), &[12, 4, 1]);
/// assert_eq!(npy.order(), npy::Order::C);
///
/// // The reader is an iterator!
/// let data: Vec<i64> = npy.collect::<Result<_, _>>()?;
/// assert_eq!(data.len(), 24);
/// # Ok(()) }
/// ```
///
/// # Migrating from `NpyData`
///
/// At construction, since `&[u8]` impls `Read`, there isn't much you have to change:
/// ```text
/// was:
///     npy::NpyData::<i64>::from_bytes(&bytes)
/// now:
///     npy::NpyReader::<i64, _>::new(&bytes[..])
/// ```
///
/// `.to_vec()` is now `.into_vec()`, consuming `self` to produce `Result<Vec<T>>`.
/// Possible errors to anticipate are `UnexpectedEof`, or, for some types, `InvalidData`.
///
/// ```text
/// was:
///     arr.to_vec()
/// now:
///     reader.into_vec().unwrap_or(|e| panic!("{}", e))
/// ```
///
/// `is_empty()` is gone due to possible ambiguity between [`len`] and [`total_len`].
/// Use the one that is appropriate for what you are doing.
///
/// FIXME TODO: Random access
///
pub struct NpyReader<T: Deserialize, R: io::Read> {
    dtype: DType,
    shape: Vec<u64>,
    strides: Vec<u64>,
    order: Order,
    n_records: u64,
    type_reader: <T as Deserialize>::TypeReader,
    // stateful parts, put together like this to remind you to always update them in sync
    reader_and_current_index: (R, u64),
}

/// Legacy type for reading `npy` files.
///
/// > This type provides the same API for reading from `npy-rs 0.4.0`, to help with migration.
/// > It will later be removed in favor of `NpyReader`.
///
/// The data is internally stored
/// as a byte array, and deserialized only on-demand to minimize unnecessary allocations.
/// The whole contents of the file can be deserialized by the [`NpyData::to_vec`] method.
#[deprecated(since = "0.5.0", note = "use NpyReader instead")]
pub struct NpyData<'a, T: Deserialize> {
    inner: NpyReader<T, &'a [u8]>,
    item_size: usize,
}

// /// This should no longer need to exist once specialization exists.
// pub struct NpyRandomAccessReader<R: io::Read, T: Deserialize> {
//     inner: NpyReader<R, T>,
// }

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

impl<R: io::Read, T: Deserialize> NpyReader<T, R> {
    /// Read the header of an `npy` file and construct an `NpyReader` for reading the data.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let (dtype, shape, order) = Self::read_and_interpret_header(&mut reader)?;
        let type_reader = match T::reader(&dtype) {
            Ok(r) => r,
            Err(e) => return Err(invalid_data(e)),
        };
        let n_records = shape.iter().product();
        let strides = strides(order, &shape);
        Ok(NpyReader {
            dtype, shape, strides, n_records, type_reader, order,
            reader_and_current_index: (reader, 0),
        })
    }

    #[inline(always)]
    fn reader(&self) -> &R {
        &self.reader_and_current_index.0
    }

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
    /// It is a function of both [`NpyReader::order`] and [`NpyReader::shape`],
    /// provided for your convenience.
    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    /// Get whether the data is in C order or fortran order.
    pub fn order(&self) -> Order {
        self.order
    }

    /// Returns the total number of records, including those that have already been read.
    /// (This is the product of [`NpyReader::shape`])
    pub fn total_len(&self) -> u64 {
        self.n_records
    }

    /// Get the remaining number of records that have not yet been read.
    pub fn len(&self) -> u64 {
        self.n_records - self.reader_and_current_index.1
    }

    /// Read all remaining, unread records into a `Vec`.
    ///
    /// This is just a convenient shorthand for `self.collect::<io::Result<Vec<_>>>()`.
    pub fn into_vec(self) -> io::Result<Vec<T>> {
        self.collect()
    }

    fn read_and_interpret_header(mut r: impl io::Read) -> io::Result<(DType, Vec<u64>, Order)> {
        let header = read_header(&mut r)?;

        let dict = match header {
            Value::Map(ref dict) => dict,
            _ => return Err(invalid_data("expected a python dict literal")),
        };

        let expect_key = |key: &str| {
            dict.get(key).ok_or_else(|| invalid_data(format_args!("dict is missing key '{}'", key)))
        };

        let order = match expect_key("fortran_order")? {
            &Value::Bool(b) => Order::from_fortran_order(b),
            _ => return Err(invalid_data(format_args!("'fortran_order' value is not a bool"))),
        };

        let shape = convert_value_to_shape(expect_key("shape")?)?;

        let descr: &Value = expect_key("descr")?;
        let dtype = DType::from_descr(descr.clone())?;
        Ok((dtype, shape, order))
    }
}

#[allow(deprecated)]
impl<'a, T: Deserialize> NpyData<'a, T> {
    /// Deserialize a NPY file represented as bytes
    pub fn from_bytes(bytes: &'a [u8]) -> io::Result<NpyData<'a, T>> {
        let inner = NpyReader::new(bytes)?;
        let item_size = inner.dtype.num_bytes();

        assert_eq!(item_size as u64 * inner.n_records, inner.reader().len() as u64);
        Ok(NpyData { inner, item_size })
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
        let item_bytes = &self.get_data_slice()[i * self.item_size..];
        self.inner.type_reader.read_one(item_bytes).unwrap()
    }

    /// Construct a vector with the deserialized contents of the whole file.
    ///
    /// The output is a flat vector with the elements in the same order that they are in the file.
    /// To help interpret the results for multidimensional data, see [`NpyData::shape`]
    /// and [`NpyData::strides`].
    pub fn to_vec(&self) -> Vec<T> {
        let mut reader = self.inner.reader().clone();
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
        if *current_index < self.n_records {
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
    use crate::out_file::to_bytes_1d;

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
        let mut reader = NpyReader::new(&bytes[..]).unwrap();

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 7);

        assert!(matches!(reader.next(), Some(Ok(100))));
        assert!(matches!(reader.next(), Some(Ok(101))));

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 5);

        assert_eq!(reader.into_vec().unwrap(), vec![102, 103, 104, 105, 106]);
    }

    #[test]
    fn test_next_after_finished_iteration() {
        let bytes = to_bytes_1d(&[100, 101, 102, 103, 104, 105, 106]).unwrap();
        let mut reader = NpyReader::<i32, _>::new(&bytes[..]).unwrap();

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 7);

        assert_eq!(reader.by_ref().count(), 7);  // run iterator to completion

        assert!(reader.next().is_none());
        assert!(reader.next().is_none());

        assert_eq!(reader.total_len(), 7);
        assert_eq!(reader.len(), 0);  // make sure this didn't underflow...
    }
}
