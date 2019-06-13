use std::io::{Result, ErrorKind, Error};

use header::{Value, DType, parse_header, convert_value_to_shape};
use serialize::{Deserialize, TypeRead};

/// The data structure representing a deserialized `npy` file.
///
/// The data is internally stored
/// as a byte array, and deserialized only on-demand to minimize unnecessary allocations.
/// The whole contents of the file can be deserialized by the [`to_vec`](#method.to_vec)
/// member function.
pub struct NpyData<'a, T: Deserialize> {
    data: &'a [u8],
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    order: Order,
    n_records: usize,
    item_size: usize,
    reader: <T as Deserialize>::Reader,
}

/// Order of axes in a file.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Order {
    /// The last dimension has a stride of 1.
    C,
    /// The first dimension has a stride of 1.
    Fortran,
}

impl Order {
    pub(crate) fn from_fortran_order(fortran_order: bool) -> Order {
        if fortran_order { Order::Fortran } else { Order::C }
    }
}

impl<'a, T: Deserialize> NpyData<'a, T> {
    /// Deserialize a NPY file represented as bytes
    pub fn from_bytes(bytes: &'a [u8]) -> ::std::io::Result<NpyData<'a, T>> {
        let (dtype, data, shape, order) = Self::get_data_slice(bytes)?;
        let reader = match T::reader(&dtype) {
            Ok(reader) => reader,
            Err(e) => return Err(Error::new(ErrorKind::InvalidData, e.to_string())),
        };
        let item_size = dtype.num_bytes();
        let n_records = shape.iter().product();
        let strides = strides(order, &shape);
        Ok(NpyData { data, dtype, shape, strides, n_records, item_size, reader, order })
    }

    /// Get the dtype as written in the file.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape as written in the file.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get strides for each of the dimensions.
    ///
    /// This is the amount by which the item index changes as you move along each dimension.
    /// It is a function of both [`NpyData::order`] and [`NpyData::shape`],
    /// provided for your convenience.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get whether the data is in C order or fortran order.
    pub fn order(&self) -> Order {
        self.order
    }

    /// Gets a single data-record with the specified flat index.
    ///
    /// Returns None if the index is out of bounds.
    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.n_records {
            Some(self.get_unchecked(i))
        } else {
            None
        }
    }

    /// Returns the total number of records
    pub fn len(&self) -> usize {
        self.n_records
    }

    /// Returns whether there are zero records in this NpyData structure.
    pub fn is_empty(&self) -> bool {
        self.n_records == 0
    }

    /// Gets a single data-record with the specified index. Panics if the index is out of bounds.
    pub fn get_unchecked(&self, i: usize) -> T {
        self.reader.read_one(&self.data[i * self.item_size..]).0
    }

    /// Construct a vector with the deserialized contents of the whole file.
    ///
    /// The output is a flat vector with the elements in the same order that they are in the file.
    /// To interpret the results for multidimensional data, see [`NpyData::shape`],
    /// and [`NpyData::fortran_order`], and [`NpyData::strides`].
    pub fn to_vec(&self) -> Vec<T> {
        let mut v = Vec::with_capacity(self.n_records);
        for i in 0..self.n_records {
            v.push(self.get_unchecked(i));
        }
        v
    }

    fn get_data_slice(bytes: &[u8]) -> Result<(DType, &[u8], Vec<usize>, Order)> {
        let (data, header) = match parse_header(bytes) {
            nom::IResult::Done(data, header) => (data, header),
            nom::IResult::Incomplete(needed) => return Err(invalid_data(format!("{:?}", needed))),
            nom::IResult::Error(err) => return Err(invalid_data(format!("{:?}", err))),
        };

        let dict = match header {
            Value::Map(ref dict) => dict,
            _ => return Err(invalid_data("expected a python dict literal")),
        };

        let expect_key = |key: &str| {
            dict.get(key).ok_or_else(|| invalid_data(format!("dict is missing key '{}'", key)))
        };

        let order = match expect_key("fortran_order")? {
            &Value::Bool(b) => Order::from_fortran_order(b),
            _ => return Err(invalid_data(format!("'fortran_order' value is not a bool"))),
        };

        let shape = {
            convert_value_to_shape(expect_key("shape")?)?
                .into_iter().map(|x: u64| x as usize).collect()
        };

        let descr: &Value = expect_key("descr")?;
        let dtype = DType::from_descr(descr.clone())?;
        Ok((dtype, data, shape, order))
    }
}

fn strides(order: Order, shape: &[usize]) -> Vec<usize> {
    match order {
        Order::C => {
            let mut strides = prefix_products(shape.iter().rev().cloned()).collect::<Vec<_>>();
            strides.reverse();
            strides
        },
        Order::Fortran => prefix_products(shape.iter().cloned()).collect(),
    }
}

fn prefix_products<I: Iterator<Item=usize>>(iter: I) -> impl Iterator<Item=usize> {
    iter.scan(1, |acc, x| { let old = *acc; *acc *= x; Some(old) })
}

fn invalid_data<S: Into<String>>(s: S) -> Error {
    Error::new(ErrorKind::InvalidData, s.into())
}

/// A result of NPY file deserialization.
///
/// It is an iterator to offer a lazy interface in case the data don't fit into memory.
pub struct IntoIter<'a, T: 'a + Deserialize> {
    data: NpyData<'a, T>,
    i: usize,
}

impl<'a, T> IntoIter<'a, T> where T: Deserialize {
    fn new(data: NpyData<'a, T>) -> Self {
        IntoIter { data, i: 0 }
    }
}

impl<'a, T: 'a> IntoIterator for NpyData<'a, T> where T: Deserialize {
    type Item = T;
    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

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

impl<'a, T> ExactSizeIterator for IntoIter<'a, T> where T: Deserialize {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strides() {
        assert_eq!(strides(Order::C, &[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(Order::C, &[]), vec![]);
        assert_eq!(strides(Order::Fortran, &[2, 3, 4]), vec![1, 2, 6]);
        assert_eq!(strides(Order::Fortran, &[]), vec![]);
    }
}
