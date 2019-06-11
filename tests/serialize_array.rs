extern crate npy;

use npy::{Deserialize, Serialize, AutoSerialize, DType, TypeStr, Field};
use npy::{TypeRead, TypeWrite};

// These tests ideally would be in npy::serialize::tests, but they require "derive"
// because arrays can only exist as record fields.

fn reader_output<T: Deserialize>(dtype: &DType, bytes: &[u8]) -> T {
    T::reader(dtype).unwrap_or_else(|e| panic!("{}", e)).read_one(bytes).0
}

fn reader_expect_err<T: Deserialize>(dtype: &DType) {
    T::reader(dtype).err().expect("reader_expect_err failed!");
}

fn writer_output<T: Serialize + ?Sized>(dtype: &DType, value: &T) -> Vec<u8> {
    let mut vec = vec![];
    T::writer(dtype).unwrap_or_else(|e| panic!("{}", e))
        .write_one(&mut vec, value).unwrap();
    vec
}

fn writer_expect_err<T: Serialize + ?Sized>(dtype: &DType) {
    T::writer(dtype).err().expect("writer_expect_err failed!");
}

#[derive(npy::Serialize, npy::Deserialize, npy::AutoSerialize)]
#[derive(Debug, PartialEq)]
struct Array3 {
    field: [i32; 3],
}

#[derive(npy::Serialize, npy::Deserialize, npy::AutoSerialize)]
#[derive(Debug, PartialEq)]
struct Array23 {
    field: [[i32; 3]; 2],
}

const ARRAY3_DESCR_LE: &str = "[('field', '<i4', (3,))]";
const ARRAY23_DESCR_LE: &str = "[('field', '<i4', (2, 3))]";

// various bad descrs for Array3
const ARRAY2_DESCR_LE: &str = "[('field', '<i4', (2,))]";
const ARRAY_SCALAR_DESCR_LE: &str = "[('field', '<i4')]";
const ARRAY_RECORD_DESCR_LE: &str = "[('field', [('lol', '<i4')])]";

#[test]
fn read_write() {
    let dtype = DType::parse(ARRAY3_DESCR_LE).unwrap();
    let value = Array3 { field: [1, 3, 5] };
    let mut bytes = vec![];
    bytes.extend_from_slice(&i32::to_le_bytes(1));
    bytes.extend_from_slice(&i32::to_le_bytes(3));
    bytes.extend_from_slice(&i32::to_le_bytes(5));

    assert_eq!(reader_output::<Array3>(&dtype, &bytes), value);
    assert_eq!(writer_output::<Array3>(&dtype, &value), bytes);
    reader_expect_err::<Array23>(&dtype);
    writer_expect_err::<Array23>(&dtype);
}

#[test]
fn read_write_nested() {
    let dtype = DType::parse(ARRAY23_DESCR_LE).unwrap();
    let value = Array23 { field: [[1, 3, 5], [7, 9, 11]] };
    let mut bytes = vec![];
    for n in vec![1, 3, 5, 7, 9, 11] {
        bytes.extend_from_slice(&i32::to_le_bytes(n));
    }

    assert_eq!(reader_output::<Array23>(&dtype, &bytes), value);
    assert_eq!(writer_output::<Array23>(&dtype, &value), bytes);
    reader_expect_err::<Array3>(&dtype);
    writer_expect_err::<Array3>(&dtype);
}

#[test]
fn incompatible() {
    // wrong size
    let dtype = DType::parse(ARRAY2_DESCR_LE).unwrap();
    writer_expect_err::<Array3>(&dtype);
    reader_expect_err::<Array3>(&dtype);

    // scalar instead of array
    let dtype = DType::parse(ARRAY_SCALAR_DESCR_LE).unwrap();
    writer_expect_err::<Array3>(&dtype);
    reader_expect_err::<Array3>(&dtype);

    // record instead of array
    let dtype = DType::parse(ARRAY_RECORD_DESCR_LE).unwrap();
    writer_expect_err::<Array3>(&dtype);
    reader_expect_err::<Array3>(&dtype);
}

#[test]
fn default_dtype() {
    let int_ty: TypeStr = {
        if 1 == i32::from_be(1) {
            ">i4".parse().unwrap()
        } else {
            "<i4".parse().unwrap()
        }
    };

    assert_eq!(Array3::default_dtype(), DType::Record(vec![
        Field {
            name: "field".to_string(),
            dtype: DType::Plain {
                ty: int_ty.clone(),
                shape: vec![3],
            },
        },
    ]));

    assert_eq!(Array23::default_dtype(), DType::Record(vec![
        Field {
            name: "field".to_string(),
            dtype: DType::Plain {
                ty: int_ty.clone(),
                shape: vec![2, 3],
            },
        },
    ]));
}
