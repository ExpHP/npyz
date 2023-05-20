use npyz::{Deserialize, Serialize, AutoSerialize, DType, TypeStr, Field};
use npyz::{TypeRead, TypeWrite};

// These tests ideally would be in npyz::serialize::tests, but they require "derive"
// because fixed-size array types can only exist as record fields.

#[cfg(target_arch="wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

fn reader_output<T: Deserialize>(dtype: &DType, bytes: &[u8]) -> T {
    T::reader(dtype).unwrap_or_else(|e| panic!("{}", e)).read_one(bytes).expect("reader_output failed")
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

#[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
#[derive(Debug, PartialEq)]
struct Array3 {
    field: [i32; 3],
}

#[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
#[derive(Debug, PartialEq)]
struct Array23 {
    field: [[i32; 3]; 2],
}

const ARRAY23_DESCR_LE: &str = "[('field', '<i4', (2, 3))]";

// good descr for Array3
const ARRAY3_DESCR_LE: &str = "[('field', '<i4', (3,))]";
// various bad descrs for Array3
const ARRAY2_DESCR_LE: &str = "[('field', '<i4', (2,))]";
const ARRAY_SCALAR_DESCR_LE: &str = "[('field', '<i4')]";
const ARRAY_RECORD_DESCR_LE: &str = "[('field', [('lol', '<i4')])]";

#[test]
fn read_write() {
    let dtype = <Array3 as npyz::AutoSerialize>::default_dtype();
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
fn read_write_explicit_dtype() {
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
            dtype: DType::Array(3, Box::new(DType::Plain(int_ty.clone()))),
        },
    ]));

    assert_eq!(Array23::default_dtype(), DType::Record(vec![
        Field {
            name: "field".to_string(),
            dtype: DType::Array(2, Box::new(DType::Array(3, Box::new(DType::Plain(int_ty.clone()))))),
        },
    ]));
}

pub mod zero_len {
    use super::*;

    #[cfg(target_arch="wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct CloseTheGap {
        left: i64,
        middle: [[[i32; 3]; 0]; 2],
        right: i32,
    }


    #[test]
    fn read_write() {
        let dtype = <CloseTheGap as npyz::AutoSerialize>::default_dtype();
        let value = CloseTheGap { left: 12, middle: [[], []], right: 5 };
        let mut bytes = vec![];
        bytes.extend_from_slice(&i64::to_le_bytes(12));
        bytes.extend_from_slice(&[]); // 'middle' should serialize to no bytes
        bytes.extend_from_slice(&i32::to_le_bytes(5));

        assert_eq!(reader_output::<CloseTheGap>(&dtype, &bytes), value);
        assert_eq!(writer_output::<CloseTheGap>(&dtype, &value), bytes);
    }
}
