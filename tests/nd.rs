use std::io::Cursor;

use npyz::{Order, WriterBuilder};

#[cfg(target_arch="wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

fn c_order_vec() -> Vec<i64> { vec![1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6] }
fn fortran_order_vec() -> Vec<i64> { vec![1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6,1,4,2,5,3,6] }

#[test]
fn read_c_order() {
    let bytes = include_bytes!("../test-data/c-order.npy");
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::C);
    assert_eq!(arr.strides(), &[12, 4, 1][..]);
    let vec = arr.into_vec::<i64>().unwrap();
    assert_eq!(vec, c_order_vec());
}

#[test]
fn read_fortran_order() {
    let bytes = include_bytes!("../test-data/f-order.npy");
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::Fortran);
    assert_eq!(arr.strides(), &[1, 2, 6][..]);
    let vec = arr.into_vec::<i64>().unwrap();
    assert_eq!(vec, fortran_order_vec());
}

#[test]
fn write_c_order() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(vec![]);
    {
        let mut npy = {
            npyz::WriteOptions::<i64>::new()
                .default_dtype()
                .shape(&[2, 3, 4])
                .writer(&mut buf)
                .begin_nd().unwrap()
        };
        npy.extend(c_order_vec()).unwrap();
        npy.finish().unwrap();
    }
    let buf = buf.into_inner();
    let arr = npyz::NpyFile::new(&buf[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::C);
    assert_eq!(arr.into_vec::<i64>().unwrap(), c_order_vec());
}

#[test]
fn write_fortran_order() {
    let mut buf: Cursor<Vec<u8>> = Cursor::new(vec![]);
    {
        let mut npy = {
            npyz::WriteOptions::<i64>::new()
                .default_dtype()
                .order(Order::Fortran)
                .shape(&[2, 3, 4])
                .writer(&mut buf)
                .begin_nd().unwrap()
        };
        npy.extend(fortran_order_vec()).unwrap();
        npy.finish().unwrap();
    }
    let buf = buf.into_inner();
    let arr = npyz::NpyFile::new(&buf[..]).unwrap();
    assert_eq!(arr.shape(), &[2, 3, 4][..]);
    assert_eq!(arr.order(), Order::Fortran);
    assert_eq!(arr.into_vec::<i64>().unwrap(), fortran_order_vec());
}
