//! Tools for reading and writing Scipy sparse matrices in NPZ format.
//!
//! _This module requires the **`"npz"`** feature.

use std::io;

use zip::read::ZipFile;

use crate::serialize::{Deserialize};
use crate::read::{Order, NpyFile};
use crate::npz::NpzArchive;

/// Raw representation of a scipy sparse matrix, in any format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sparse<T> {
    /// The matrix is in COOrdinate format.
    #[allow(missing_docs)] Coo(Coo<T>),
    /// The matrix is in Compressed Sparse Row format.
    #[allow(missing_docs)] Csr(Csr<T>),
    /// The matrix is in Compressed Sparse Column format.
    #[allow(missing_docs)] Csc(Csc<T>),
    /// The matrix is in DIAgonal format.
    #[allow(missing_docs)] Dia(Dia<T>),
    /// The matrix is in Block Sparse Row format.
    #[allow(missing_docs)] Bsr(Bsr<T>),
}

/// Raw representation of a [`scipy.sparse.coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coo<T> {
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// A vector of length `nnz` containing all of the stored elements.
    pub data: Vec<T>,
    /// A vector of length `nnz` indicating the row of each element.
    pub row: Vec<u64>,
    /// A vector of length `nnz` indicating the column of each element.
    pub col: Vec<u64>,
}

/// Raw representation of a [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Csr<T> {
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// A vector of length `nnz` containing all of the nonzero elements, sorted by row.
    pub data: Vec<T>,
    /// A vector of length `nnz` indicating the column of each element.
    ///
    /// > Beware: scipy **does not** require or guarantee that the column indices within each row are sorted.
    pub indices: Vec<u64>,
    /// A vector of length `nrow + 1` indicating the indices that partition [`data`]
    /// and [`indices`] into data for each row.
    ///
    /// Typically, the elements are nondecreasing, with the first equal to 0 and the final equal
    /// to `nnz` (though the set of requirements that are actually *validated* by scipy are
    /// weaker and somewhat arbitrary).
    pub indptr: Vec<usize>,
}

/// Raw representation of a [`scipy.sparse.csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Csc<T> {
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// A vector of length `nnz` containing all of the nonzero elements, sorted by column.
    pub data: Vec<T>,
    /// A vector of length `nnz` indicating the row of each element.
    ///
    /// > Beware: scipy **does not** require or guarantee that the row indices within each column are sorted.
    pub indices: Vec<u64>,
    /// A vector of length `ncol + 1` indicating the indices that partition [`data`]
    /// and [`indices`] into data for each column.
    ///
    /// Typically, the elements are nondecreasing, with the first equal to 0 and the final equal
    /// to `nnz` (though the set of requirements that are actually *validated* by scipy are
    /// weaker and somewhat arbitrary).
    pub indptr: Vec<usize>,
}

/// Raw representation of a [`scipy.sparse.dia_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dia<T> {
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// Contains the C-order data of a shape `[nnzd, length]` ndarray.
    ///
    /// Scipy's own documentation is lackluster, but the value of `length` appears to be any
    /// value `0 <= length <= ncol` and is typically 1 greater than the rightmost column that
    /// contains a nonzero entry.  The values in each diagonal appear to be stored at an index
    /// equal to their column.
    pub data: Vec<T>,
    /// A vector of length `nnzd` indicating which diagonal is stored in each row of `data`.
    ///
    /// Negative offsets are below the main diagonal.  Offsets can appear in any order.
    pub offsets: Vec<i64>,
}

/// Raw representation of a [`scipy.sparse.bsr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bsr<T> {
    /// Dimensions of the matrix `[nrow, ncol]`.
    ///
    /// These dimensions must be divisible by the respective elements of `blocksize`.
    pub shape: [u64; 2],
    /// Size of the blocks in the matrix.
    pub blocksize: [usize; 2],

    /// Contains the C-order data of a shape `[nnzb, block_nrow, block_ncol]` ndarray.
    ///
    /// (effectively concatenating the flattened data of all nonzero blocks, sorted by superrow)
    pub data: Vec<T>,
    /// A vector of length `nnzb` indicating the supercolumn index of each block.
    ///
    /// > Beware: scipy **does not** require or guarantee that the column indices within each row are sorted.
    pub indices: Vec<u64>,
    /// A vector of length `(nrow / block_nrow) + 1` indicating the indices which partition
    /// [`indices`] and the outermost axis of [`data`] into data for each superrow.
    ///
    /// Typically, the elements are nondecreasing, with the first equal to 0 and the final equal
    /// to `nnzb` (though the set of requirements that are actually *validated* by scipy are
    /// weaker and somewhat arbitrary).
    pub indptr: Vec<usize>,
}

impl<T: Deserialize> Sparse<T> {
    /// Read a sparse matrix saved by `scipy.sparse.save_npz`.
    pub fn from_npz<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>) -> io::Result<Self> {
        let format = extract_scalar::<Vec<u8>, _>(npz, "format")?;

        match &format[..] {
            b"coo" => Ok(Sparse::Coo(Coo::from_npz(npz)?)),
            b"csc" => Ok(Sparse::Csc(Csc::from_npz(npz)?)),
            b"csr" => Ok(Sparse::Csr(Csr::from_npz(npz)?)),
            b"dia" => Ok(Sparse::Dia(Dia::from_npz(npz)?)),
            b"bsr" => Ok(Sparse::Bsr(Bsr::from_npz(npz)?)),
            _ => Err(invalid_data(format_args!("bad format: {}", show_format(&format[..])))),
        }
    }
}

impl<T: Deserialize> Coo<T> {
    /// Read a sparse `coo_matrix` saved by `scipy.sparse.save_npz`.
    pub fn from_npz<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>) -> io::Result<Self> {
        expect_format(npz, "coo")?;
        let shape = extract_shape(npz, "shape")?;
        let row = extract_indices(npz, "row")?;
        let col = extract_indices(npz, "col")?;
        let data = extract_1d::<T, _>(npz, "data")?;
        Ok(Coo { data, shape, row, col })
    }
}

impl<T: Deserialize> Csr<T> {
    /// Read a sparse `csr_matrix` saved by `scipy.sparse.save_npz`.
    pub fn from_npz<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>) -> io::Result<Self> {
        expect_format(npz, "csr")?;
        let shape = extract_shape(npz, "shape")?;
        let indices = extract_indices(npz, "indices")?;
        let indptr = extract_usize_indices(npz, "indptr")?;
        let data = extract_1d::<T, _>(npz, "data")?;
        Ok(Csr { data, shape, indices, indptr })
    }
}

impl<T: Deserialize> Csc<T> {
    /// Read a sparse `csc_matrix` saved by `scipy.sparse.save_npz`.
    pub fn from_npz<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>) -> io::Result<Self> {
        expect_format(npz, "csc")?;
        let shape = extract_shape(npz, "shape")?;
        let indices = extract_indices(npz, "indices")?;
        let indptr = extract_usize_indices(npz, "indptr")?;
        let data = extract_1d::<T, _>(npz, "data")?;
        Ok(Csc { data, shape, indices, indptr })
    }
}

impl<T: Deserialize> Dia<T> {
    /// Read a sparse `dia_matrix` saved by `scipy.sparse.save_npz`.
    pub fn from_npz<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>) -> io::Result<Self> {
        expect_format(npz, "dia")?;
        let shape = extract_shape(npz, "shape")?;
        let offsets = extract_signed_indices(npz, "offsets")?;
        let (data, _) = extract_nd::<T, _>(npz, "data", 2)?;
        Ok(Dia { data, shape, offsets })
    }
}

impl<T: Deserialize> Bsr<T> {
    /// Read a sparse `bsr_matrix` saved by `scipy.sparse.save_npz`.
    pub fn from_npz<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>) -> io::Result<Self> {
        expect_format(npz, "bsr")?;
        let shape = extract_shape(npz, "shape")?;
        let indices = extract_indices(npz, "indices")?;
        let indptr = extract_usize_indices(npz, "indptr")?;
        let (data, data_shape) = extract_nd::<T, _>(npz, "data", 3)?;
        let blocksize = [data_shape[1], data_shape[2]];
        Ok(Bsr { data, shape, indices, indptr, blocksize })
    }
}

fn show_format(format: &[u8]) -> String {
    let str = format.iter().map(|&b| match b {
        // ASCII printable
        0x20..=0x7f => std::str::from_utf8(&[b]).unwrap().to_string(),
        _ => format!("\\x{:02X}", b),
    }).collect::<Vec<_>>().join("");

    format!("'{}'", str)
}

fn expect_format<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, expected: &str) -> io::Result<()> {
    let format: Vec<u8> = extract_scalar(npz, "format")?;
    if format != expected.as_bytes() {
        return Err(invalid_data(format_args!("wrong format: expected '{}', got {}", expected, show_format(&format))))
    }
    Ok(())
}

fn extract_scalar<T: Deserialize, R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, name: &str) -> io::Result<T> {
    let npy = extract_and_check_ndim(npz, name, 0)?;
    Ok(npy.into_vec::<T>()?.into_iter().next().expect("scalar so must have 1 elem"))
}

fn extract_shape<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, name: &str) -> io::Result<[u64; 2]> {
    let shape = extract_indices(npz, name)?;
    if shape.len() != 2 {
        return Err(invalid_data(format_args!("invalid length for '{}' (got {}, expected 2)", name, shape.len())))
    }
    Ok([shape[0], shape[1]])
}

fn extract_usize_indices<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, name: &str) -> io::Result<Vec<usize>> {
    Ok(extract_indices(npz, name)?.into_iter().map(|x| x as usize).collect())
}

// Read indices from npz which may be i32 or i64, but are nonnegative.
// FIXME: in the future we may allow automatic widening during deserialization, in which case
//        this can be simplified extract_1d::<u64>
fn extract_indices<R: io::Read + io::Seek>(npz: &mut NpzArchive<R, >, name: &str) -> io::Result<Vec<u64>> {
    let npy = extract_and_check_ndim(npz, name, 1)?;
    match npy.try_data::<i32>() {
        Ok(data) => data.map(|result| result.map(|x| x as u64)).collect(),
        Err(npy) => match npy.try_data::<i64>() {
            Ok(data) => data.map(|result| result.map(|x| x as u64)).collect(),
            Err(npy) => Err(invalid_data(format_args!("invalid dtype for '{}' in sparse matrix: {}", name, npy.dtype().descr()))),
        },
    }
}

// Read indices from npz which may be i32 or i64.
// FIXME: in the future we may allow automatic widening during deserialization, in which case
//        this can be replaced with extract_1d::<i64>
fn extract_signed_indices<R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, name: &str) -> io::Result<Vec<i64>> {
    let npy = extract_and_check_ndim(npz, name, 1)?;
    match npy.try_data::<i32>() {
        Ok(data) => data.map(|result| result.map(|x| x as i64)).collect(),
        Err(npy) => match npy.try_data::<i64>() {
            Ok(data) => data.collect(),
            Err(npy) => Err(invalid_data(format_args!("invalid dtype for '{}' in sparse matrix: {}", name, npy.dtype().descr()))),
        },
    }
}

fn extract_1d<T: Deserialize, R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, name: &str) -> io::Result<Vec<T>> {
    let npy = extract_and_check_ndim(npz, name, 1)?;
    npy.into_vec::<T>()
}

fn extract_nd<T: Deserialize, R: io::Read + io::Seek>(npz: &mut NpzArchive<R>, name: &str, expected_ndim: usize) -> io::Result<(Vec<T>, Vec<usize>)> {
    let npy = extract_and_check_ndim(npz, name, expected_ndim)?;
    if npy.order() != Order::C {
        return Err(invalid_data(format_args!("fortran order is not currently supported for array '{}' in sparse NPZ file", name)));
    }
    let shape = npy.shape().iter().map(|&x| x as usize).collect();
    let data = npy.into_vec::<T>()?;
    Ok((data, shape))
}

fn extract_and_check_ndim<'a, R: io::Read + io::Seek>(npz: &'a mut NpzArchive<R>, name: &str, expected_ndim: usize) -> io::Result<NpyFile<ZipFile<'a>>> {
    let npy = npz.by_name(name)?.ok_or_else(|| invalid_data(format_args!("missing array '{}' from sparse array", name)))?;
    let ndim = npy.shape().len();
    if ndim != expected_ndim {
        return Err(invalid_data(format_args!("invalid ndim for {}: {} (expected {})", name, ndim, expected_ndim)));
    }
    Ok(npy)
}

fn invalid_data<S: ToString>(s: S) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.to_string())
}
