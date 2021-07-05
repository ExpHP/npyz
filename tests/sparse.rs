use npyz::npz::NpzArchive;
use npyz::sparse;

fn open_test_npz(name: &str) -> NpzArchive<std::io::BufReader<std::fs::File>> {
    NpzArchive::open(format!("test-data/{}", name)).unwrap()
}

// Python code to create NPZs:
//
//   import numpy as np
//   import scipy.sparse as ss
//   m = np.array([[1, 0, 4, 0, 0, 0], [0, 2, 0, 0, 0, 0], [6, 0, 7, 0, 0, 0]])
//   ss.save_npz('test-data/sparse-coo.npz', ss.coo_matrix(m))
//   ss.save_npz('test-data/sparse-csr.npz', ss.csr_matrix(m))
//   ss.save_npz('test-data/sparse-csc.npz', ss.csc_matrix(m))
//   ss.save_npz('test-data/sparse-dia.npz', ss.dia_matrix(m))
//   ss.save_npz('test-data/sparse-bsr.npz', ss.bsr_matrix(m, blocksize=(1,2)))

#[test]
fn read_sparse_coo() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("sparse-coo.npz")).unwrap();
    assert_eq!(m, sparse::Coo {
        shape: [3, 6],
        data: vec![1, 4, 2, 6, 7],
        row: vec![0, 0, 1, 2, 2],
        col: vec![0, 2, 1, 0, 2],
    });
}

#[test]
fn read_sparse_csr() {
    let m = sparse::Csr::<i64>::from_npz(&mut open_test_npz("sparse-csr.npz")).unwrap();
    assert_eq!(m, sparse::Csr {
        shape: [3, 6],
        data: vec![1, 4, 2, 6, 7],
        indices: vec![0, 2, 1, 0, 2],
        indptr: vec![0, 2, 3, 5],
    });
}

#[test]
fn read_sparse_csc() {
    let m = sparse::Csc::<i64>::from_npz(&mut open_test_npz("sparse-csc.npz")).unwrap();
    assert_eq!(m, sparse::Csc {
        shape: [3, 6],
        data: vec![1, 6, 2, 4, 7],
        indices: vec![0, 2, 1, 0, 2],
        indptr: vec![0, 2, 3, 5, 5, 5, 5],
    });
}

#[test]
fn read_sparse_dia() {
    let m = sparse::Dia::<i64>::from_npz(&mut open_test_npz("sparse-dia.npz")).unwrap();
    assert_eq!(m, sparse::Dia {
        shape: [3, 6],
        offsets: vec![-2, 0, 2],
        data: vec![
            6, 0, 0,
            1, 2, 7,
            0, 0, 4,
        ],
    });
}

#[test]
fn read_sparse_bsr() {
    let m = sparse::Bsr::<i64>::from_npz(&mut open_test_npz("sparse-bsr.npz")).unwrap();
    assert_eq!(m, sparse::Bsr {
        shape: [3, 6],
        blocksize: [1, 2],
        data: vec![
            1, 0, 4, 0,
            0, 2,
            6, 0, 7, 0,
        ],
        indices: vec![0, 1, 0, 0, 1],
        indptr: vec![0, 2, 3, 5],
    });
}

#[test]
fn read_sparse_dynamic() {
    use sparse::Sparse;

    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("sparse-coo.npz")).unwrap(), Sparse::Coo(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("sparse-csr.npz")).unwrap(), Sparse::Csr(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("sparse-csc.npz")).unwrap(), Sparse::Csc(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("sparse-bsr.npz")).unwrap(), Sparse::Bsr(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("sparse-dia.npz")).unwrap(), Sparse::Dia(_)));
}

#[test]
fn read_wrong_format_err() {
    sparse::Csr::<i64>::from_npz(&mut open_test_npz("sparse-coo.npz")).unwrap_err();
}

#[test]
fn read_sparse_with_long_indices() {
    // - test successful read of i64 indices in CSR
    unimplemented!()
}

#[test]
fn read_sparse_dia_with_long_offsets() {
    // - test successful read of i64 offsets in DIA
    unimplemented!()
}

#[test]
fn read_sparse_coo_with_dupes() {
    // - test successful read of COO with duplicates
    unimplemented!()
}

#[test]
fn read_sparse_csr_unsorted() {
    // - test successful read of CSR with unsorted columns
    unimplemented!()
}

#[test]
fn read_fortran_order_err() {
    // - fortran order error
    unimplemented!()
}

#[test]
fn read_bad_dimension_err() {
    // - error when data of bsr is not 3d
    unimplemented!()
}
