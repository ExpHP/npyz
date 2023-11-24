// Example demonstrating how you could use the `sprs` crate to work with sparse matrices.

use npyz::{AutoSerialize, Deserialize};

// npyz::sparse uses u64 for indices and usize for indptr
type MyCsMat<T> = sprs::CsMatI<T, u64, usize>;
type MyCsMatView<'a, T> = sprs::CsMatViewI<'a, T, u64, usize>;

// Read a sprs CSR matrix from file
fn load_sprs_csr<T>(path: &std::path::Path) -> Result<MyCsMat<T>, Box<dyn std::error::Error>>
where
    T: Deserialize + Clone,
{
    let mut npz = npyz::npz::NpzArchive::open(path)?;
    let csr = npyz::sparse::Csr::from_npz(&mut npz)?;
    let npyz::sparse::Csr {
        shape,
        indices,
        indptr,
        data,
    } = csr;

    // IMPORTANT:  Scipy CSR matrices might not be sorted so use new_from_unsorted!
    let sprs = MyCsMat::new_from_unsorted(
        (shape[0] as usize, shape[1] as usize),
        indptr,
        indices,
        data,
    )
    .map_err(|(_, _, _, e)| e)?;
    Ok(sprs)
}

// Save a sprs CSR matrix to file
fn save_sprs_csr<T>(
    path: &std::path::Path,
    sprs: &MyCsMatView<'_, T>,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: AutoSerialize + Clone,
{
    assert!(sprs.is_csr());

    let indptr = sprs.indptr();
    let csr = npyz::sparse::CsrBase {
        shape: [sprs.rows() as u64, sprs.cols() as u64],
        indices: sprs.indices(),
        indptr: indptr.to_proper(),
        data: sprs.data(),
    };
    let mut npz = npyz::npz::NpzWriter::create(path)?;
    csr.write_npz(&mut npz)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sprs = load_sprs_csr::<i64>("test-data/sparse/csr.npz".as_ref())?;

    println!("{:?}", sprs.to_dense());

    save_sprs_csr("examples/output/sparse-sprs.npz".as_ref(), &sprs.view())
}
