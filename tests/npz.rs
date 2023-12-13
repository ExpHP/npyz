use std::io;
use npyz::WriterBuilder;
use npyz::npz::{NpzArchive, NpzWriter};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[test]
fn read_uncompressed() {
    let mut data = io::Cursor::new(include_bytes!("../test-data/uncompressed.npz").to_vec());
    test_basic_read(NpzArchive::new(&mut data).unwrap())
}
#[test]
fn read_compressed() {
    let mut data = io::Cursor::new(include_bytes!("../test-data/compressed.npz").to_vec());
    test_basic_read(NpzArchive::new(&mut data).unwrap())
}

// Python code to create NPZs:
//   import numpy as np
//   np.savez('test-data/uncompressed.npz', ints=np.array([1,2,3,4]), floats=np.array([[1.0], [2.0]]))
//   np.savez_compressed('test-data/compressed.npz', ints=np.array([1,2,3,4]), floats=np.array([[1.0], [2.0]]))
fn test_basic_read(mut npz: NpzArchive<impl io::Read + io::Seek>) {
    let mut names = npz.array_names().collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["floats", "ints"]);

    let ints = npz.by_name("ints").expect("error!?").expect("missing?!");
    assert_eq!(ints.shape(), &[4]);
    assert_eq!(ints.into_vec::<i64>().unwrap(), vec![1, 2, 3, 4]);

    let floats = npz.by_name("floats").expect("error!?").expect("missing?!");
    assert_eq!(floats.shape(), &[2, 1]);
    assert_eq!(floats.into_vec::<f64>().unwrap(), vec![1.0, 2.0]);

    assert!(matches!(npz.by_name("non-existent"), Ok(None)));
}

#[test]
fn basic_write() {
    let mut buf = io::Cursor::new(vec![]);
    let mut npz = NpzWriter::new(&mut buf);

    npz.array("ints", Default::default()).unwrap()
        .default_dtype()
        .shape(&[4])
        .begin_nd().unwrap()
        .extend(vec![1_i64, 2, 3, 4]).unwrap();

    npz.array("floats", Default::default()).unwrap()
        .default_dtype()
        .shape(&[2, 1])
        .begin_nd().unwrap()
        .extend(vec![1.0, 2.0]).unwrap();

    drop(npz); // release borrow

    // Check that the file can be read back
    let bytes = buf.into_inner();
    test_basic_read(NpzArchive::new(io::Cursor::new(&bytes[..])).unwrap());
}
