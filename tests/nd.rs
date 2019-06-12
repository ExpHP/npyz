extern crate npy;

#[test]
fn c_order() {
    let bytes = std::fs::read("tests/c-order.npy").unwrap();
    let arr = npy::NpyData::<i64>::from_bytes(&bytes).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), npy::Order::C);
    assert_eq!(arr.to_vec(), vec![1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6]);
    assert_eq!(arr.strides(), &[12, 4, 1][..]);
    assert_eq!(arr.get(1), Some(1));
    assert_eq!(arr.get(4), Some(2));
    assert_eq!(arr.get(12), Some(4));
    assert_eq!(arr.get(2*3*4 - 1), Some(6));
    assert_eq!(arr.get(2*3*4), None);
}

#[test]
fn fortran_order() {
    let bytes = std::fs::read("tests/f-order.npy").unwrap();
    let arr = npy::NpyData::<i64>::from_bytes(&bytes).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), npy::Order::Fortran);
    assert_eq!(arr.to_vec(), vec![1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6]);
    assert_eq!(arr.strides(), &[1, 2, 6][..]);
    assert_eq!(arr.get(6), Some(1));
    assert_eq!(arr.get(2), Some(2));
    assert_eq!(arr.get(1), Some(4));
    assert_eq!(arr.get(2*3*4 - 1), Some(6));
    assert_eq!(arr.get(2*3*4), None);
}
