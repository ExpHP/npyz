use std::io::Cursor;

use nippy::Order;

fn c_order_vec() -> Vec<i64> { vec![1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6] }
fn fortran_order_vec() -> Vec<i64> { vec![1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6] }

#[test]
fn read_c_order() {
    let bytes = std::fs::read("tests/c-order.npy").unwrap();
    let arr = nippy::NpyReader::<i64, _>::new(&bytes[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::C);
    assert_eq!(arr.strides(), &[12, 4, 1][..]);
    let vec = arr.into_vec().unwrap();
    assert_eq!(vec, c_order_vec());
}

#[test]
fn read_fortran_order() {
    let bytes = std::fs::read("tests/f-order.npy").unwrap();
    let arr = nippy::NpyReader::<i64, _>::new(&bytes[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::Fortran);
    assert_eq!(arr.strides(), &[1, 2, 6][..]);
    let vec = arr.into_vec().unwrap();
    assert_eq!(vec, fortran_order_vec());
}

#[test]
fn write_c_order() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(vec![]);
    {
        let mut npy = {
            nippy::Builder::<i64>::new()
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
    let arr = nippy::NpyReader::<i64, _>::new(&buf[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::C);
    assert_eq!(arr.into_vec().unwrap(), c_order_vec());
}

#[test]
fn write_fortran_order() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(vec![]);
    {
        let mut npy = {
            nippy::Builder::<i64>::new()
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
    let arr = nippy::NpyReader::<i64, _>::new(&buf[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::Fortran);
    assert_eq!(arr.into_vec().unwrap(), fortran_order_vec());
}
