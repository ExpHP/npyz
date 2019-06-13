extern crate npy;
use npy::Order;
use std::io::Cursor;

fn c_order_vec() -> Vec<i64> { vec![1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6] }
fn fortran_order_vec() -> Vec<i64> { vec![1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6] }

#[test]
fn read_c_order() {
    let bytes = std::fs::read("tests/c-order.npy").unwrap();
    let arr = npy::NpyData::<i64>::from_bytes(&bytes).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::C);
    assert_eq!(arr.to_vec(), c_order_vec());
    assert_eq!(arr.strides(), &[12, 4, 1][..]);
    assert_eq!(arr.get(1), Some(1));
    assert_eq!(arr.get(4), Some(2));
    assert_eq!(arr.get(12), Some(4));
    assert_eq!(arr.get(2*3*4 - 1), Some(6));
    assert_eq!(arr.get(2*3*4), None);
}

#[test]
fn read_fortran_order() {
    let bytes = std::fs::read("tests/f-order.npy").unwrap();
    let arr = npy::NpyData::<i64>::from_bytes(&bytes).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::Fortran);
    assert_eq!(arr.to_vec(), fortran_order_vec());
    assert_eq!(arr.strides(), &[1, 2, 6][..]);
    assert_eq!(arr.get(6), Some(1));
    assert_eq!(arr.get(2), Some(2));
    assert_eq!(arr.get(1), Some(4));
    assert_eq!(arr.get(2*3*4 - 1), Some(6));
    assert_eq!(arr.get(2*3*4), None);
}

#[test]
fn write_c_order() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(vec![]);
    {
        let mut npy = {
            npy::Builder::<i64>::new()
                .default_dtype()
                .begin_nd(&mut buf, &[2, 3, 4])
                .unwrap()
        };
        for x in c_order_vec() {
            npy.push(&x).unwrap();
        }
        npy.finish().unwrap();
    }
    let buf = buf.into_inner();
    let arr = npy::NpyData::<i64>::from_bytes(&buf).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::C);
    assert_eq!(arr.to_vec(), c_order_vec());
}

#[test]
fn write_fortran_order() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(vec![]);
    {
        let mut npy = {
            npy::Builder::<i64>::new()
                .default_dtype()
                .order(Order::Fortran)
                .begin_nd(&mut buf, &[2, 3, 4])
                .unwrap()
        };
        for x in fortran_order_vec() {
            npy.push(&x).unwrap();
        }
        npy.finish().unwrap();
    }
    let buf = buf.into_inner();
    let arr = npy::NpyData::<i64>::from_bytes(&buf).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::Fortran);
    assert_eq!(arr.to_vec(), fortran_order_vec());
}
