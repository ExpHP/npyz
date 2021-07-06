//! Tools for reading and writing Scipy sparse matrices in NPZ format.
//!
//! ```rust
//! use std::io;
//! use npyz::sparse;
//!
//! fn main() -> io::Result<()> {
//!     let mut npz = npyz::npz::NpzArchive::open("test-data/sparse/csr.npz")?;
//!     let mat = sparse::Csr::<i64>::from_npz(&mut npz)?;
//!
//!     // sparse matrices have public fields named after the attributes found in scipy.
//!     // read or manipulate them however you like!
//!     let sparse::Csr { data, indices, indptr, shape } = &mat;
//!     println!("Shape: {:?}", shape);
//!     println!("Indices: {:?}", indices);
//!     println!("Indptr: {:?}", indptr);
//!     println!("Data: {:?}", data);
//!
//!     // write to any io::Write
//!     let writer = io::BufWriter::new(std::fs::File::create("examples/output/sparse-doctest.npz")?);
//!     mat.write_npz(&mut npyz::npz::NpzWriter::new(writer))?;
//!     Ok(())
//! }
//! ```
//!
//! No methods are provided on these types beyond reading and writing.  If you want to do sparse
//! matrix math, then you should use the data you have read to construct a matrix type from a
//! dedicated sparse matrix library.
//!
//! For instance, an example of how to use this module to save and load CSR matrices from the
//! [`sprs`](https://crates.io/crates/sprs) crate can be found
//! [in the examples directory](https://github.com/ExpHP/npyz/tree/master/examples).
//!
//! _This module requires the **`"npz"`** feature._

use std::io;
use std::ops::Deref;

use zip::read::ZipFile;

use crate::serialize::{Deserialize, AutoSerialize};
use crate::read::{Order, NpyFile};
use crate::write::{WriterBuilder};
use crate::npz::{NpzArchive, NpzWriter};
use crate::header::DType;

// =============================================================================
// Types

/// Raw representation of a scipy sparse matrix whose exact format is known at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseBase<T, Data, Indices, Indptr, Offsets>
where  // note: explicit 'where' makes rustdoc less intimidating
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
    Offsets: AsRef<[i64]>,
{
    /// The matrix is in COOrdinate format.
    Coo(CooBase<T, Data, Indices>),
    /// The matrix is in Compressed Sparse Row format.
    Csr(CsrBase<T, Data, Indices, Indptr>),
    /// The matrix is in Compressed Sparse Column format.
    Csc(CscBase<T, Data, Indices, Indptr>),
    /// The matrix is in DIAgonal format.
    Dia(DiaBase<T, Data, Offsets>),
    /// The matrix is in Block Sparse Row format.
    Bsr(BsrBase<T, Data, Indices, Indptr>),
}

/// A sparse matrix (of type known at runtime) that owns its data.
///
/// This is an enum.  Please consult [`SparseBase`] to see the list of variants.
pub type Sparse<T> = SparseBase<T, Vec<T>, Vec<u64>, Vec<usize>, Vec<i64>>;

/// Raw representation of a [`scipy.sparse.coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html).
///
/// In spirit, each field is simply a Vec. (see the type alias [`Coo`]).
/// This generic base class exists in order to allow you to use slices when writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooBase<T, Data, Indices>
where  // note: explicit 'where' makes rustdoc less intimidating
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
{
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// A vector of length `nnz` containing all of the stored elements.
    pub data: Data,
    /// A vector of length `nnz` indicating the row of each element.
    pub row: Indices,
    /// A vector of length `nnz` indicating the column of each element.
    pub col: Indices,
}

/// A COO matrix that owns its data.
///
/// Please consult the documentation of [`CooBase`] to see the list of fields publicly available on this type.
pub type Coo<T> = CooBase<T, Vec<T>, Vec<u64>>;

/// Raw representation of a [`scipy.sparse.csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).
///
/// In spirit, each field is simply a Vec. (see the type alias [`Csr`]).
/// This generic base class exists in order to allow you to use slices when writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsrBase<T, Data, Indices, Indptr>
where  // note: explicit 'where' makes rustdoc less intimidating
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
{
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// A vector of length `nnz` containing all of the nonzero elements, sorted by row.
    pub data: Data,
    /// A vector of length `nnz` indicating the column of each element.
    ///
    /// **Beware:** scipy **does not** require or guarantee that the column indices within each row are sorted.
    pub indices: Indices,
    /// A vector of length `nrow + 1` indicating the indices that partition [`Self::data`]
    /// and [`Self::indices`] into data for each row.
    ///
    /// Typically, the elements are nondecreasing, with the first equal to 0 and the final equal
    /// to `nnz` (though the set of requirements that are actually *validated* by scipy are
    /// weaker and somewhat arbitrary).
    pub indptr: Indptr,
}

/// A CSR matrix that owns its data.
///
/// Please consult the documentation of [`CsrBase`] to see the list of fields publicly available on this type.
pub type Csr<T> = CsrBase<T, Vec<T>, Vec<u64>, Vec<usize>>;

/// Raw representation of a [`scipy.sparse.csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html).
///
/// In spirit, each field is simply a Vec. (see the type alias [`Csc`]).
/// This generic base class exists in order to allow you to use slices when writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscBase<T, Data, Indices, Indptr>
where  // note: explicit 'where' makes rustdoc less intimidating
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
{
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// A vector of length `nnz` containing all of the nonzero elements, sorted by column.
    pub data: Data,
    /// A vector of length `nnz` indicating the row of each element.
    ///
    /// **Beware:** scipy **does not** require or guarantee that the row indices within each column are sorted.
    pub indices: Indices,
    /// A vector of length `ncol + 1` indicating the indices that partition [`Self::data`]
    /// and [`Self::indices`] into data for each column.
    ///
    /// Typically, the elements are nondecreasing, with the first equal to 0 and the final equal
    /// to `nnz` (though the set of requirements that are actually *validated* by scipy are
    /// weaker and somewhat arbitrary).
    pub indptr: Indptr,
}

/// A CSC matrix that owns its data.
///
/// Please consult the documentation of [`CscBase`] to see the list of fields publicly available on this type.
pub type Csc<T> = CscBase<T, Vec<T>, Vec<u64>, Vec<usize>>;

/// Raw representation of a [`scipy.sparse.dia_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html).
///
/// In spirit, each field is simply a Vec. (see the type alias [`Dia`]).
/// This generic base class exists in order to allow you to use slices when writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiaBase<T, Data, Offsets>
where  // note: explicit 'where' makes rustdoc less intimidating
    Data: Deref<Target=[T]>,
    Offsets: AsRef<[i64]>,
{
    /// Dimensions of the matrix `[nrow, ncol]`.
    pub shape: [u64; 2],
    /// Contains the C-order data of a shape `[nnzd, length]` ndarray.
    ///
    /// Scipy's own documentation is lackluster, but the value of `length` appears to be any
    /// value `0 <= length <= ncol` and is typically 1 greater than the rightmost column that
    /// contains a nonzero entry.  The values in each diagonal appear to be stored at an index
    /// equal to their column.
    pub data: Data,
    /// A vector of length `nnzd` indicating which diagonal is stored in each row of `data`.
    ///
    /// Negative offsets are below the main diagonal.  Offsets can appear in any order.
    pub offsets: Offsets,
}

/// A DIA matrix that owns its data.
///
/// Please consult the documentation of [`DiaBase`] to see the list of fields publicly available on this type.
pub type Dia<T> = DiaBase<T, Vec<T>, Vec<i64>>;

/// Raw representation of a [`scipy.sparse.bsr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html).
///
/// In spirit, each field is simply a Vec. (see the type alias [`Bsr`]).
/// This generic base class exists in order to allow you to use slices when writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BsrBase<T, Data, Indices, Indptr>
where  // note: explicit 'where' makes rustdoc less intimidating
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
{
    /// Dimensions of the matrix `[nrow, ncol]`.
    ///
    /// These dimensions must be divisible by the respective elements of `blocksize`.
    pub shape: [u64; 2],
    /// Size of the blocks in the matrix.
    pub blocksize: [usize; 2],

    /// Contains the C-order data of a shape `[nnzb, block_nrow, block_ncol]` ndarray.
    ///
    /// (effectively concatenating the flattened data of all nonzero blocks, sorted by superrow)
    pub data: Data,
    /// A vector of length `nnzb` indicating the supercolumn index of each block.
    ///
    /// **Beware:** scipy **does not** require or guarantee that the column indices within each row are sorted.
    pub indices: Indices,
    /// A vector of length `(nrow / block_nrow) + 1` indicating the indices which partition
    /// [`Self::indices`] and the outermost axis of [`Self::data`] into data for each superrow.
    ///
    /// Typically, the elements are nondecreasing, with the first equal to 0 and the final equal
    /// to `nnzb` (though the set of requirements that are actually *validated* by scipy are
    /// weaker and somewhat arbitrary).
    pub indptr: Indptr,
}

/// A BSR matrix that owns its data.
///
/// Please consult the documentation of [`BsrBase`] to see the list of fields publicly available on this type.
pub type Bsr<T> = BsrBase<T, Vec<T>, Vec<u64>, Vec<usize>>;

// =============================================================================
// Reading

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

// -----

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

// =============================================================================
// Writing

impl<T, Data, Indices, Indptr, Offsets> SparseBase<T, Data, Indices, Indptr, Offsets>
where
    T: AutoSerialize,
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
    Offsets: AsRef<[i64]>
{
    /// Write a sparse matrix, like `scipy.sparse.save_npz`.
    pub fn write_npz<W: io::Write + io::Seek>(&self, npz: &mut NpzWriter<W>) -> io::Result<()> {
        match self {
            SparseBase::Coo(m) => m.write_npz(npz),
            SparseBase::Csc(m) => m.write_npz(npz),
            SparseBase::Csr(m) => m.write_npz(npz),
            SparseBase::Dia(m) => m.write_npz(npz),
            SparseBase::Bsr(m) => m.write_npz(npz),
        }
    }
}

impl<T, Data, Indices> CooBase<T, Data, Indices>
where
    T: AutoSerialize,
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
{
    /// Write a sparse `coo_matrix` matrix, like `scipy.sparse.save_npz`.
    ///
    /// # Panics
    ///
    /// This method does not currently perform any significant validation of input,
    /// but validation (with panics) may be added later in a future semver major bump.
    pub fn write_npz<W: io::Write + io::Seek>(&self, npz: &mut NpzWriter<W>) -> io::Result<()> {
        let CooBase { data, shape, row, col } = self;
        write_format(npz, "coo")?;
        write_shape(npz, shape)?;
        write_indices(npz, "row", row.as_ref().iter().map(|&x| x as i64))?;
        write_indices(npz, "col", col.as_ref().iter().map(|&x| x as i64))?;
        write_data(npz, &data, &[data.len() as u64])?;
        Ok(())
    }
}

impl<T, Data, Indices, Indptr> CsrBase<T, Data, Indices, Indptr>
where
    T: AutoSerialize,
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
{
    /// Write a sparse `csr_matrix` matrix, like `scipy.sparse.save_npz`.
    ///
    /// # Panics
    ///
    /// This method does not currently perform any significant validation of input,
    /// but validation (with panics) may be added later in a future semver major bump.
    pub fn write_npz<W: io::Write + io::Seek>(&self, npz: &mut NpzWriter<W>) -> io::Result<()> {
        let CsrBase { data, shape, indices, indptr } = self;
        write_format(npz, "csr")?;
        write_shape(npz, shape)?;
        write_indices(npz, "indices", indices.as_ref().iter().map(|&x| x as i64))?;
        write_indices(npz, "indptr", indptr.as_ref().iter().map(|&x| x as i64))?;
        write_data(npz, &data, &[data.len() as u64])?;
        Ok(())
    }
}

impl<T, Data, Indices, Indptr> CscBase<T, Data, Indices, Indptr>
where
    T: AutoSerialize,
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
{
    /// Write a sparse `csc_matrix` matrix, like `scipy.sparse.save_npz`.
    ///
    /// # Panics
    ///
    /// This method does not currently perform any significant validation of input,
    /// but validation (with panics) may be added later in a future semver major bump.
    pub fn write_npz<W: io::Write + io::Seek>(&self, npz: &mut NpzWriter<W>) -> io::Result<()> {
        let CscBase { data, shape, indices, indptr } = self;
        write_format(npz, "csc")?;
        write_shape(npz, shape)?;
        write_indices(npz, "indices", indices.as_ref().iter().map(|&x| x as i64))?;
        write_indices(npz, "indptr", indptr.as_ref().iter().map(|&x| x as i64))?;
        write_data(npz, &data, &[data.len() as u64])?;
        Ok(())
    }
}

impl<T, Data, Offsets> DiaBase<T, Data, Offsets>
where
    T: AutoSerialize,
    Data: Deref<Target=[T]>,
    Offsets: AsRef<[i64]>,
{
    /// Write a sparse `dia_matrix` matrix, like `scipy.sparse.save_npz`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `offsets.len()`.
    pub fn write_npz<W: io::Write + io::Seek>(&self, npz: &mut NpzWriter<W>) -> io::Result<()> {
        let DiaBase { data, shape, offsets } = self;
        write_format(npz, "dia")?;
        write_shape(npz, shape)?;
        write_indices(npz, "offsets", offsets.as_ref().iter().copied())?;

        let num_offsets = offsets.as_ref().len();
        assert_eq!(data.len() % num_offsets, 0);
        let length = data.len() / num_offsets;
        write_data(npz, &data, &[length as u64, num_offsets as u64])?;
        Ok(())
    }
}

impl<T, Data, Indices, Indptr> BsrBase<T, Data, Indices, Indptr>
where
    T: AutoSerialize,
    Data: Deref<Target=[T]>,
    Indices: AsRef<[u64]>,
    Indptr: AsRef<[usize]>,
{
    /// Write a sparse `bsr_matrix` matrix, like `scipy.sparse.save_npz`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not equal to `indices.len() * blocksize[0] * blocksize[1]`.
    pub fn write_npz<W: io::Write + io::Seek>(&self, npz: &mut NpzWriter<W>) -> io::Result<()> {
        let BsrBase { data, shape, indices, indptr, blocksize } = self;
        write_format(npz, "bsr")?;
        write_shape(npz, shape)?;
        write_indices(npz, "indices", indices.as_ref().iter().map(|&x| x as i64))?;
        write_indices(npz, "indptr", indptr.as_ref().iter().map(|&x| x as i64))?;

        assert_eq!(data.len(), indices.as_ref().len() * blocksize[0] * blocksize[1]);
        write_data(npz, &data, &[indices.as_ref().len() as u64, blocksize[0] as u64, blocksize[1] as u64])?;
        Ok(())
    }
}

// -----

fn zip_file_options() -> zip::write::FileOptions {
    Default::default()
}

fn write_format<W: io::Write + io::Seek>(npz: &mut NpzWriter<W>, format: &str) -> io::Result<()> {
    npz.array("format", zip_file_options())?
        .dtype(DType::Plain("|S3".parse().unwrap()))
        .shape(&[])
        .begin_nd()?
        .push(format.as_bytes())
}

fn write_shape<W: io::Write + io::Seek>(npz: &mut NpzWriter<W>, shape: &[u64]) -> io::Result<()> {
    assert_eq!(shape.len(), 2);
    npz.array("shape", zip_file_options())?
        .default_dtype()
        .shape(&[2])
        .begin_nd()?
        .extend(shape.iter().map(|&x| x as i64))
}

// Write signed ints as either i32 or i64 depending on their max value.
fn write_indices<W: io::Write + io::Seek>(npz: &mut NpzWriter<W>, name: &str, data: impl ExactSizeIterator<Item=i64> + Clone) -> io::Result<()> {
    let (min, max) = most_negative_and_positive(data.clone());
    if (i32::MIN as i64) <= min && max <= (i32::MAX as i64) {
        // small indices
        npz.array(name, zip_file_options())?
            .default_dtype()
            .shape(&[data.len() as u64])
            .begin_nd()?
            .extend(data.map(|x| x as i32))
    } else {
        // long indices
        npz.array(name, zip_file_options())?
            .default_dtype()
            .shape(&[data.len() as u64])
            .begin_nd()?
            .extend(data)
    }
}

fn most_negative_and_positive(data: impl ExactSizeIterator<Item=i64>) -> (i64, i64) {
    let mut best_negative = 0;
    let mut best_positive = 0;
    // single pass for better memory characteristics
    for x in data {
        best_negative = best_negative.min(x);
        best_positive = best_positive.max(x);
    }
    (best_negative, best_positive)
}

fn write_data<W: io::Write + io::Seek, T: AutoSerialize>(npz: &mut NpzWriter<W>, data: &[T], shape: &[u64]) -> io::Result<()> {
    npz.array("data", zip_file_options())?
        .default_dtype()
        .shape(shape)
        .begin_nd()?
        .extend(data)
}
