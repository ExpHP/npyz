use npyz::npz::NpzArchive;
use npyz::sparse;

fn open_test_npz(name: &str) -> NpzArchive<std::io::BufReader<std::fs::File>> {
    NpzArchive::open(format!("test-data/sparse/{}", name)).unwrap()
}

// Python code to create NPZs:
//
//   import numpy as np
//   import scipy.sparse as ss
//   m = np.array([[1, 0, 4, 0, 0, 0], [0, 2, 0, 0, 0, 0], [6, 0, 7, 0, 0, 0]])
//   ss.save_npz('test-data/sparse/coo.npz', ss.coo_matrix(m))
//   ss.save_npz('test-data/sparse/csr.npz', ss.csr_matrix(m))
//   ss.save_npz('test-data/sparse/csc.npz', ss.csc_matrix(m))
//   ss.save_npz('test-data/sparse/dia.npz', ss.dia_matrix(m))
//   ss.save_npz('test-data/sparse/bsr.npz', ss.bsr_matrix(m, blocksize=(1,2)))

#[test]
fn read_sparse_coo() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("coo.npz")).unwrap();
    assert_eq!(m, sparse::Coo {
        shape: [3, 6],
        data: vec![1, 4, 2, 6, 7],
        row: vec![0, 0, 1, 2, 2],
        col: vec![0, 2, 1, 0, 2],
    });
}

#[test]
fn read_sparse_csr() {
    let m = sparse::Csr::<i64>::from_npz(&mut open_test_npz("csr.npz")).unwrap();
    assert_eq!(m, sparse::Csr {
        shape: [3, 6],
        data: vec![1, 4, 2, 6, 7],
        indices: vec![0, 2, 1, 0, 2],
        indptr: vec![0, 2, 3, 5],
    });
}

#[test]
fn read_sparse_csc() {
    let m = sparse::Csc::<i64>::from_npz(&mut open_test_npz("csc.npz")).unwrap();
    assert_eq!(m, sparse::Csc {
        shape: [3, 6],
        data: vec![1, 6, 2, 4, 7],
        indices: vec![0, 2, 1, 0, 2],
        indptr: vec![0, 2, 3, 5, 5, 5, 5],
    });
}

#[test]
fn read_sparse_dia() {
    let m = sparse::Dia::<i64>::from_npz(&mut open_test_npz("dia.npz")).unwrap();
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
    let m = sparse::Bsr::<i64>::from_npz(&mut open_test_npz("bsr.npz")).unwrap();
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

    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("coo.npz")).unwrap(), Sparse::Coo(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("csr.npz")).unwrap(), Sparse::Csr(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("csc.npz")).unwrap(), Sparse::Csc(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("bsr.npz")).unwrap(), Sparse::Bsr(_)));
    assert!(matches!(Sparse::<i64>::from_npz(&mut open_test_npz("dia.npz")).unwrap(), Sparse::Dia(_)));
}

#[test]
fn read_wrong_format_err() {
    let err = sparse::Csr::<i64>::from_npz(&mut open_test_npz("coo.npz")).unwrap_err();
    assert!(err.to_string().contains("format"));
}

#[test]
fn read_sparse_with_long_indices() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("coo-long.npz")).unwrap();
    assert_eq!(m, sparse::Coo {
        shape: [0x8_0000_0000, 0x10_0000_0000],
        data: vec![3, 1],
        row: vec![0x4_0000_0000, 3],
        col: vec![4, 0x2_0000_0000],
    });
}

#[test]
fn read_sparse_dia_with_long_offsets() {
    let m = sparse::Dia::<i64>::from_npz(&mut open_test_npz("dia-long.npz")).unwrap();
    assert_eq!(m, sparse::Dia {
        shape: [0x8_0000_0000, 2],
        offsets: vec![-0x8_0000_0000 + 3],
        data: vec![0, 1],
    });
}

#[test]
fn read_sparse_coo_with_dupes() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("coo-dupes.npz")).unwrap();
    assert_eq!(m, sparse::Coo {
        shape: [5, 5],
        data: vec![10, 20],
        row: vec![2, 2],
        col: vec![3, 3],
    });
}

#[test]
fn read_sparse_csr_unsorted() {
    // python:
    //   import scipy.sparse as ss
    //   m=ss.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    //   ss.save_npz('test-data/sparse/csr-unsorted.npz', m @ m)
    let m = sparse::Csr::<i64>::from_npz(&mut open_test_npz("csr-unsorted.npz")).unwrap();
    assert_eq!(m, sparse::Csr {
        shape: [3, 3],
        data: vec![2, 2, 1, 2, 2],
        indices: vec![2, 0, 1, 2, 0],
        indptr: vec![0, 2, 3, 5],
    });
}

#[test]
fn read_fortran_order_err() {
    // python:
    //   import numpy as np
    //   npz = np.load('test-data/sparse/bsr.npz')
    //   mats = dict(**npz)
    //
    //   assert not mats['data'].flags['F_CONTIGUOUS']
    //   mats['data'] = mats['data'].T.copy().T
    //   assert mats['data'].flags['F_CONTIGUOUS']
    //
    //   np.savez('test-data/sparse/bsr-f-order.npz', **mats)
    let err = sparse::Bsr::<i64>::from_npz(&mut open_test_npz("bsr-f-order.npz")).unwrap_err();
    assert!(err.to_string().contains("ortran"));
}

#[test]
fn read_bad_dimension_err() {
    // python:
    //   import numpy as np
    //   npz = np.load('test-data/sparse/bsr.npz')
    //   mats = dict(**npz)
    //   mats['data'] = mats['data'].reshape((-1, 2))
    //   np.savez('test-data/sparse/bsr-bad-ndim.npz', **mats)
    let err = sparse::Bsr::<i64>::from_npz(&mut open_test_npz("bsr-bad-ndim.npz")).unwrap_err();
    assert!(err.to_string().contains("ndim"));
}
