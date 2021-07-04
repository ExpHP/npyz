#[test]
fn read_uncompressed() {
    test_basic_npz("test-data/uncompressed.npz");
}
#[test]
fn read_compressed() {
    test_basic_npz("test-data/compressed.npz");
}

// Python code to create NPZs:
//   import numpy as np
//   np.savez('test-data/uncompressed.npz', ints=np.array([1,2,3,4]), floats=np.array([[1.0], [2.0]]))
//   np.savez_compressed('test-data/compressed.npz', ints=np.array([1,2,3,4]), floats=np.array([[1.0], [2.0]]))
fn test_basic_npz(path: &str) {
    let mut npz = npyz::npz::NpzArchive::open(path).unwrap();

    let mut names = npz.array_names().collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["floats", "ints"]);

    let ints = npz.by_name::<i64>("ints").expect("error!?").expect("missing?!");
    assert_eq!(ints.shape(), &[4]);
    assert_eq!(ints.into_vec().unwrap(), vec![1, 2, 3, 4]);

    let floats = npz.by_name::<f64>("floats").expect("error!?").expect("missing?!");
    assert_eq!(floats.shape(), &[2, 1]);
    assert_eq!(floats.into_vec().unwrap(), vec![1.0, 2.0]);

    assert!(matches!(npz.by_name::<i32>("non-existent"), Ok(None)));
}
