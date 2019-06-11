extern crate npy;
extern crate byteorder;

use byteorder::ByteOrder;
use std::io::{Read, Write};
use byteorder::{WriteBytesExt, LittleEndian};
use npy::{DType, Field, OutFile, Serialize, Deserialize, AutoSerialize};

#[derive(Serialize, Deserialize, AutoSerialize)]
#[derive(Debug, PartialEq, Clone)]
struct Nested {
    v1: f32,
    v2: f32,
}

#[derive(Serialize, Deserialize, AutoSerialize)]
#[derive(Debug, PartialEq, Clone)]
struct Array {
    v_i8: i8,
    v_i16: i16,
    v_i32: i32,
    v_i64: i64,
    v_u8: u8,
    v_u16: u16,
    v_u32: u32,
    v_u64: u64,
    v_f32: f32,
    v_f64: f64,
    v_arr_u32: [u32;7],
    v_mat_u64: [[u64; 3]; 5],
    vec: Vector5,
    nested: Nested,
}

#[derive(Debug, PartialEq, Clone)]
struct Vector5(Vec<i32>);

impl AutoSerialize for Vector5 {
    #[inline]
    fn default_dtype() -> DType {
        DType::Plain { ty: "<i4".parse().unwrap(), shape: vec![5] }
    }
}

impl Serialize for Vector5 {
    type Writer = Vector5Writer;

    fn writer(dtype: &DType) -> Result<Self::Writer, npy::DTypeError> {
        if dtype == &Self::default_dtype() {
            Ok(Vector5Writer)
        } else {
            Err(npy::DTypeError::custom("Vector5 only supports '<i4' format!"))
        }
    }
}

impl Deserialize for Vector5 {
    type Reader = Vector5Reader;

    fn reader(dtype: &DType) -> Result<Self::Reader, npy::DTypeError> {
        if dtype == &Self::default_dtype() {
            Ok(Vector5Reader)
        } else {
            Err(npy::DTypeError::custom("Vector5 only supports '<i4' format!"))
        }
    }
}

struct Vector5Writer;
struct Vector5Reader;

impl npy::TypeWrite for Vector5Writer {
    type Value = Vector5;

    #[inline]
    fn write_one<W: Write>(&self, mut writer: W, value: &Self::Value) -> std::io::Result<()> {
        for i in 0..5 {
            writer.write_i32::<LittleEndian>(value.0[i])?
        }
        Ok(())
    }
}

impl npy::TypeRead for Vector5Reader {
    type Value = Vector5;

    #[inline]
    fn read_one<'a>(&self, mut remainder: &'a [u8]) -> (Self::Value, &'a [u8]) {
        let mut ret = Vector5(vec![]);
        for _ in 0..5 {
            ret.0.push(LittleEndian::read_i32(remainder));
            remainder = &remainder[4..];
        }
        (ret, remainder)
    }
}

#[test]
fn roundtrip() {
    let n = 100i64;

    let mut arrays = vec![];
    for i in 0..n {
        let j = i as u32 * 5 + 2;
        let k = i as u64 * 2 + 5;
        let a = Array {
            v_i8: i as i8,
            v_i16: i as i16,
            v_i32: i as i32,
            v_i64: i as i64,
            v_u8: i as u8,
            v_u16: i as u16,
            v_u32: i as u32,
            v_u64: i as u64,
            v_f32: i as f32,
            v_f64: i as f64,
            v_arr_u32: [j,1+j,2+j,3+j,4+j,5+j,6+j],
            v_mat_u64: [[k,1+k,2+k],[3+k,4+k,5+k],[6+k,7+k,8+k],[9+k,10+k,11+k],[12+k,13+k,14+k]],
            vec: Vector5(vec![1,2,3,4,5]),
            nested: Nested { v1: 10.0 * i as f32, v2: i as f32 },
        };
        arrays.push(a);
    }

    npy::to_file("tests/roundtrip.npy", arrays.clone()).unwrap();

    let mut buf = vec![];
    std::fs::File::open("tests/roundtrip.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let arrays2 = npy::NpyData::from_bytes(&buf).unwrap().to_vec();
    assert_eq!(arrays, arrays2);
}

fn plain_field(name: &str, dtype: &str) -> Field {
    Field {
        name: name.to_string(),
        dtype: DType::new_scalar(dtype.parse().unwrap()),
    }
}

#[test]
fn roundtrip_with_plain_dtype() {
    let array_written = vec![2., 3., 4., 5.];

    npy::to_file("tests/roundtrip_plain.npy", array_written.clone()).unwrap();

    let mut buffer = vec![];
    std::fs::File::open("tests/roundtrip_plain.npy").unwrap()
        .read_to_end(&mut buffer).unwrap();

    let array_read = npy::NpyData::from_bytes(&buffer).unwrap().to_vec();
    assert_eq!(array_written, array_read);
}

#[test]
fn roundtrip_byteorder() {
    let path = "tests/roundtrip_byteorder.npy";

    #[derive(npy::Serialize, npy::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        be_u32: u32,
        le_u32: u32,
        be_f32: f32,
        le_f32: f32,
        be_i8: i8,
        le_i8: i8,
        na_i8: i8,
    }

    let dtype = DType::Record(vec![
        plain_field("be_u32", ">u4"),
        plain_field("le_u32", "<u4"),
        plain_field("be_f32", ">f4"),
        plain_field("le_f32", "<f4"),
        // check that all byteorders are legal for i1
        plain_field("be_i8", ">i1"),
        plain_field("le_i8", "<i1"),
        plain_field("na_i8", "|i1"),
    ]);

    let row = Row {
        be_u32: 0x01_02_03_04,
        le_u32: 0x01_02_03_04,
        be_f32: -6259853398707798016.0, // 0xdeadbeef
        le_f32: -6259853398707798016.0,
        be_i8: 5,
        le_i8: 6,
        na_i8: 7,
    };

    let expected_data_bytes = {
        let mut buf = vec![];
        buf.extend_from_slice(b"\x01\x02\x03\x04\x04\x03\x02\x01");
        buf.extend_from_slice(b"\xDE\xAD\xBE\xEF\xEF\xBE\xAD\xDE");
        buf.extend_from_slice(b"\x05\x06\x07");
        buf
    };

    let mut out_file = OutFile::open_with_dtype(&dtype, path).unwrap();
    out_file.push(&row).unwrap();
    out_file.close().unwrap();

    // Make sure it actually wrote in the correct byteorders.
    let buffer = std::fs::read(path).unwrap();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npy::NpyData::<Row>::from_bytes(&buffer).unwrap();
    assert_eq!(data.to_vec(), vec![row]);
    assert_eq!(data.dtype(), dtype);
}

#[test]
fn roundtrip_datetime() {
    let path = "tests/roundtrip_datetime.npy";

    // Similar to:
    //
    // ```
    // import numpy.datetime64 as dt
    // import numpy as np
    //
    // arr = np.array([(
    //     dt('2011-01-01', 'ns'),
    //     dt('2011-01-02') - dt('2011-01-01'),
    //     dt('2011-01-02') - dt('2011-01-01'),
    // )], dtype=[
    //     ('datetime', '<M8[ns]'),
    //     ('timedelta_le', '<m8[D]'),
    //     ('timedelta_be', '>m8[D]'),
    // ])
    // ```
    #[derive(npy::Serialize, npy::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        datetime: u64,
        timedelta_le: i64,
        timedelta_be: i64,
    }

    let dtype = DType::Record(vec![
        plain_field("datetime", "<M8[ns]"),
        plain_field("timedelta_le", "<m8[D]"),
        plain_field("timedelta_be", ">m8[D]"),
    ]);

    let row = Row {
        datetime: 1_293_840_000_000_000_000,
        timedelta_le: 1,
        timedelta_be: 1,
    };

    let expected_data_bytes = {
        let mut buf = vec![];
        buf.extend_from_slice(&i64::to_le_bytes(1_293_840_000_000_000_000));
        buf.extend_from_slice(&i64::to_le_bytes(1));
        buf.extend_from_slice(&i64::to_be_bytes(1));
        buf
    };

    let mut out_file = OutFile::open_with_dtype(&dtype, path).unwrap();
    out_file.push(&row).unwrap();
    out_file.close().unwrap();

    let buffer = std::fs::read(path).unwrap();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npy::NpyData::<Row>::from_bytes(&buffer).unwrap();
    assert_eq!(data.to_vec(), vec![row]);
    assert_eq!(data.dtype(), dtype);
}

#[test]
fn roundtrip_bytes() {
    let path = "tests/roundtrip_bytes.npy";

    // Similar to:
    //
    // ```
    // import numpy as np
    //
    // arr = np.array([(
    //     b"\x00such\x00wow",
    //     b"\x00such\x00wow\x00\x00\x00",
    // )], dtype=[
    //     ('bytestr', '|S12'),
    //     ('raw', '|V12'),
    // ])
    // ```
    #[derive(npy::Serialize, npy::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        bytestr: Vec<u8>,
        raw: Vec<u8>,
    }

    let dtype = DType::Record(vec![
        plain_field("bytestr", "|S12"),
        plain_field("raw", "|V12"),
    ]);

    let row = Row {
        // checks that:
        // * bytestr can be shorter than the len
        // * bytestr can contain non-trailing NULs
        bytestr: b"\x00lol\x00lol".to_vec(),
        // * raw can contain trailing NULs
        raw: b"\x00lol\x00lol\x00\x00\x00\x00".to_vec(),
    };

    let expected_data_bytes = {
        let mut buf = vec![];
        // check that bytestr is nul-padded
        buf.extend_from_slice(b"\x00lol\x00lol\x00\x00\x00\x00");
        buf.extend_from_slice(b"\x00lol\x00lol\x00\x00\x00\x00");
        buf
    };

    let mut out_file = OutFile::open_with_dtype(&dtype, path).unwrap();
    out_file.push(&row).unwrap();
    out_file.close().unwrap();

    let buffer = std::fs::read(path).unwrap();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npy::NpyData::<Row>::from_bytes(&buffer).unwrap();
    assert_eq!(data.to_vec(), vec![row]);
    assert_eq!(data.dtype(), dtype);
}

// check that all byte orders are identical for bytestrings
// (i.e. don't accidentally reverse the bytestrings)
#[test]
fn roundtrip_bytes_byteorder() {
    let path = "tests/roundtrip_bytes_byteorder.npy";

    #[derive(npy::Serialize, npy::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        s_le: Vec<u8>,
        s_be: Vec<u8>,
        s_na: Vec<u8>,
        v_le: Vec<u8>,
        v_be: Vec<u8>,
        v_na: Vec<u8>,
    };

    let dtype = DType::Record(vec![
        plain_field("s_le", "<S4"),
        plain_field("s_be", ">S4"),
        plain_field("s_na", "|S4"),
        plain_field("v_le", "<V4"),
        plain_field("v_be", ">V4"),
        plain_field("v_na", "|V4"),
    ]);

    let row = Row {
        s_le: b"abcd".to_vec(),
        s_be: b"abcd".to_vec(),
        s_na: b"abcd".to_vec(),
        v_le: b"abcd".to_vec(),
        v_be: b"abcd".to_vec(),
        v_na: b"abcd".to_vec(),
    };

    let expected_data_bytes = {
        let mut buf = vec![];
        for _ in 0..6 {
            buf.extend_from_slice(b"abcd");
        }
        buf
    };

    let mut out_file = OutFile::open_with_dtype(&dtype, path).unwrap();
    out_file.push(&row).unwrap();
    out_file.close().unwrap();

    let buffer = std::fs::read(path).unwrap();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npy::NpyData::<Row>::from_bytes(&buffer).unwrap();
    assert_eq!(data.to_vec(), vec![row]);
    assert_eq!(data.dtype(), dtype);
}
