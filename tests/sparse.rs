use npyz::npz::{NpzArchive, NpzWriter};
use npyz::sparse;

fn open_test_npz(name: &str) -> NpzArchive<std::io::BufReader<std::fs::File>> {
    NpzArchive::open(format!("test-data/sparse/{}", name)).unwrap()
}

macro_rules! test_writing_sparse {
    ($Ty:ty, $matrix:expr) => {{
        let matrix = $matrix;
        let mut buf = std::io::Cursor::new(vec![]);
        matrix.write_npz(&mut NpzWriter::new(&mut buf)).unwrap();

        let bytes = buf.into_inner();
        let mut read_npz = NpzArchive::new(std::io::Cursor::new(&bytes)).unwrap();
        let read_matrix = <$Ty>::from_npz(&mut read_npz).unwrap();
        assert_eq!(read_matrix, matrix);
    }};
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

// matrices created by the above code
fn example_coo() -> sparse::Coo<i64> {
    sparse::Coo {
        shape: [3, 6],
        data: vec![1, 4, 2, 6, 7],
        row: vec![0, 0, 1, 2, 2],
        col: vec![0, 2, 1, 0, 2],
    }
}
fn example_csr() -> sparse::Csr<i64> {
    sparse::Csr {
        shape: [3, 6],
        data: vec![1, 4, 2, 6, 7],
        indices: vec![0, 2, 1, 0, 2],
        indptr: vec![0, 2, 3, 5],
    }
}
fn example_csc() -> sparse::Csc<i64> {
    sparse::Csc {
        shape: [3, 6],
        data: vec![1, 6, 2, 4, 7],
        indices: vec![0, 2, 1, 0, 2],
        indptr: vec![0, 2, 3, 5, 5, 5, 5],
    }
}
fn example_dia() -> sparse::Dia<i64> {
    sparse::Dia {
        shape: [3, 6],
        offsets: vec![-2, 0, 2],
        data: vec![
            6, 0, 0,
            1, 2, 7,
            0, 0, 4,
        ],
    }
}
fn example_bsr() -> sparse::Bsr<i64> {
    sparse::Bsr {
        shape: [3, 6],
        blocksize: [1, 2],
        data: vec![
            1, 0, 4, 0,
            0, 2,
            6, 0, 7, 0,
        ],
        indices: vec![0, 1, 0, 0, 1],
        indptr: vec![0, 2, 3, 5],
    }
}

// Contents of coo-long.npz
fn example_coo_long() -> sparse::Coo<i64> {
    sparse::Coo {
        shape: [0x8_0000_0000, 0x10_0000_0000],
        data: vec![3, 1],
        row: vec![0x4_0000_0000, 3],
        col: vec![4, 0x2_0000_0000],
    }
}

// Contents of dia-long.npz
fn example_dia_long() -> sparse::Dia<i64> {
    sparse::Dia {
        shape: [0x8_0000_0000, 2],
        offsets: vec![-0x8_0000_0000 + 3],
        data: vec![0, 1],
    }
}

// Contents of coo-dupes.npz
fn example_coo_dupes() -> sparse::Coo<i64> {
    sparse::Coo {
        shape: [5, 5],
        data: vec![10, 20],
        row: vec![2, 2],
        col: vec![3, 3],
    }
}

fn example_csr_unsorted() -> sparse::Csr<i64> {
    // python:
    //   import scipy.sparse as ss
    //   m=ss.csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    //   ss.save_npz('test-data/sparse/csr-unsorted.npz', m @ m)
    sparse::Csr {
        shape: [3, 3],
        data: vec![2, 2, 1, 2, 2],
        indices: vec![2, 0, 1, 2, 0],
        indptr: vec![0, 2, 3, 5],
    }
}

#[test]
fn read_sparse_coo() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("coo.npz")).unwrap();
    assert_eq!(m, example_coo());
}

#[test]
fn read_sparse_csr() {
    let m = sparse::Csr::<i64>::from_npz(&mut open_test_npz("csr.npz")).unwrap();
    assert_eq!(m, example_csr());
}

#[test]
fn read_sparse_csc() {
    let m = sparse::Csc::<i64>::from_npz(&mut open_test_npz("csc.npz")).unwrap();
    assert_eq!(m, example_csc());
}

#[test]
fn read_sparse_dia() {
    let m = sparse::Dia::<i64>::from_npz(&mut open_test_npz("dia.npz")).unwrap();
    assert_eq!(m, example_dia());
}

#[test]
fn read_sparse_bsr() {
    let m = sparse::Bsr::<i64>::from_npz(&mut open_test_npz("bsr.npz")).unwrap();
    assert_eq!(m, example_bsr());
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

#[test] fn write_sparse_coo() { test_writing_sparse!(sparse::Coo<i64>, example_coo()) }
#[test] fn write_sparse_csr() { test_writing_sparse!(sparse::Csr<i64>, example_csr()) }
#[test] fn write_sparse_csc() { test_writing_sparse!(sparse::Csc<i64>, example_csc()) }
#[test] fn write_sparse_bsr() { test_writing_sparse!(sparse::Bsr<i64>, example_bsr()) }
#[test] fn write_sparse_dia() { test_writing_sparse!(sparse::Dia<i64>, example_dia()) }

#[test] fn write_sparse_dynamic() {
    use sparse::Sparse;

    test_writing_sparse!(Sparse<i64>, Sparse::Csr(example_csr()));
    test_writing_sparse!(Sparse<i64>, Sparse::Csc(example_csc()));
    test_writing_sparse!(Sparse<i64>, Sparse::Coo(example_coo()));
    test_writing_sparse!(Sparse<i64>, Sparse::Dia(example_dia()));
    test_writing_sparse!(Sparse<i64>, Sparse::Bsr(example_bsr()));
}


#[test]
fn read_wrong_format_err() {
    let err = sparse::Csr::<i64>::from_npz(&mut open_test_npz("coo.npz")).unwrap_err();
    assert!(err.to_string().contains("format"));
}

#[test]
fn sparse_with_long_indices() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("coo-long.npz")).unwrap();
    assert_eq!(m, example_coo_long());

    test_writing_sparse!(sparse::Coo<i64>, m);
}

#[test]
fn sparse_dia_with_long_offsets() {
    let m = sparse::Dia::<i64>::from_npz(&mut open_test_npz("dia-long.npz")).unwrap();
    assert_eq!(m, example_dia_long());

    test_writing_sparse!(sparse::Dia<i64>, m);
}

#[test]
fn sparse_coo_with_dupes() {
    let m = sparse::Coo::<i64>::from_npz(&mut open_test_npz("coo-dupes.npz")).unwrap();
    assert_eq!(m, example_coo_dupes());

    test_writing_sparse!(sparse::Coo<i64>, m);
}

#[test]
fn sparse_csr_unsorted() {
    let m = sparse::Csr::<i64>::from_npz(&mut open_test_npz("csr-unsorted.npz")).unwrap();
    assert_eq!(m, example_csr_unsorted());

    test_writing_sparse!(sparse::Csr<i64>, m);
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
