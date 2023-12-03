// * The definition of a "mixed script confusable" can change over time, so when testing unicode
//   identifiers there is no safe identifier we can use that is guaranteed never to generate the warning.
// * It has to be allowed at global level for the crate, because that's the level at
//   which the warning is generated.
#![allow(mixed_script_confusables)]

use std::io::{self, Read, Write, Cursor};
use byteorder::{WriteBytesExt, ReadBytesExt, LittleEndian};
use npyz::{DType, Field, Serialize, Deserialize, AutoSerialize, WriterBuilder};
use half::f16;

// Allows to use the `#[test]` on WASM.
#[cfg(target_arch="wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

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
    v_f16: f16,
    v_f32: f32,
    v_f64: f64,
    v_arr_u32: [u32;7],
    v_mat_u64: [[u64; 3]; 5],
    vec: Vector5,
    nested: Nested,
}

#[derive(Serialize, Deserialize, AutoSerialize)]
#[derive(Debug, PartialEq, Clone)]
struct Version3 {
    v1: f32,
    v2: f32,
}

#[derive(Debug, PartialEq, Clone)]
struct Vector5(Vec<i32>);

impl AutoSerialize for Vector5 {
    #[inline]
    fn default_dtype() -> DType {
        DType::Array(5, Box::new(DType::Plain("<i4".parse().unwrap())))
    }
}

impl Serialize for Vector5 {
    type TypeWriter = Vector5Writer;

    fn writer(dtype: &DType) -> Result<Self::TypeWriter, npyz::DTypeError> {
        if dtype == &Self::default_dtype() {
            Ok(Vector5Writer)
        } else {
            Err(npyz::DTypeError::custom("Vector5 only supports '<i4' format!"))
        }
    }
}

impl Deserialize for Vector5 {
    type TypeReader = Vector5Reader;

    fn reader(dtype: &DType) -> Result<Self::TypeReader, npyz::DTypeError> {
        if dtype == &Self::default_dtype() {
            Ok(Vector5Reader)
        } else {
            Err(npyz::DTypeError::custom("Vector5 only supports '<i4' format!"))
        }
    }
}

struct Vector5Writer;
struct Vector5Reader;

impl npyz::TypeWrite for Vector5Writer {
    type Value = Vector5;

    #[inline]
    fn write_one<W: Write>(&self, mut writer: W, value: &Self::Value) -> std::io::Result<()> {
        for i in 0..5 {
            writer.write_i32::<LittleEndian>(value.0[i])?
        }
        Ok(())
    }
}

impl npyz::TypeRead for Vector5Reader {
    type Value = Vector5;

    #[inline]
    fn read_one<R: Read>(&self, mut reader: R) -> std::io::Result<Self::Value> {
        let mut ret = Vector5(vec![]);
        for _ in 0..5 {
            ret.0.push(reader.read_i32::<LittleEndian>()?);
        }
        Ok(ret)
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
            v_f16: f16::from_f32(i as f32),
            v_f32: i as f32,
            v_f64: i as f64,
            v_arr_u32: [j,1+j,2+j,3+j,4+j,5+j,6+j],
            v_mat_u64: [[k,1+k,2+k],[3+k,4+k,5+k],[6+k,7+k,8+k],[9+k,10+k,11+k],[12+k,13+k,14+k]],
            vec: Vector5(vec![1,2,3,4,5]),
            nested: Nested { v1: 10.0 * i as f32, v2: i as f32 },
        };
        arrays.push(a);
    }

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().default_dtype().writer(&mut writer).begin_1d().unwrap();
    out_file.extend(arrays.iter()).unwrap();
    out_file.finish().unwrap();

    let buf = writer.into_inner();

    assert_version(&buf, (1, 0));

    let arrays2 = npyz::NpyFile::new(&buf[..]).unwrap().into_vec().unwrap();
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

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().default_dtype().writer(&mut writer).begin_1d().unwrap();
    out_file.extend(array_written.iter()).unwrap();
    out_file.finish().unwrap();

    let buffer = writer.into_inner();

    let array_read = npyz::NpyFile::new(&buffer[..]).unwrap().into_vec().unwrap();
    assert_eq!(array_written, array_read);
}

#[test]
fn roundtrip_byteorder() {
    #[derive(npyz::Serialize, npyz::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        be_u32: u32,
        le_u32: u32,
        be_f16: f16,
        le_f16: f16,
        be_f32: f32,
        le_f32: f32,
        be_i8: i8,
        le_i8: i8,
        na_i8: i8,
    }

    let dtype = DType::Record(vec![
        plain_field("be_u32", ">u4"),
        plain_field("le_u32", "<u4"),
        plain_field("be_f16", ">f2"),
        plain_field("le_f16", "<f2"),
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
        be_f16: f16::from_f32_const(-123456789.0),
        le_f16: f16::from_f32_const(-123456789.0),
        be_f32: -6259853398707798016.0, // 0xdeadbeef
        le_f32: -6259853398707798016.0,
        be_i8: 5,
        le_i8: 6,
        na_i8: 7,
    };

    let expected_data_bytes = {
        let mut buf = vec![];
        buf.extend_from_slice(b"\x01\x02\x03\x04\x04\x03\x02\x01");
        buf.extend_from_slice(b"\xFC\x00\x00\xFC");
        buf.extend_from_slice(b"\xDE\xAD\xBE\xEF\xEF\xBE\xAD\xDE");
        buf.extend_from_slice(b"\x05\x06\x07");
        buf
    };

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().dtype(dtype.clone()).writer(&mut writer).begin_1d().unwrap();
    out_file.push(&row).unwrap();
    out_file.finish().unwrap();

    // Make sure it actually wrote in the correct byteorders.
    let buffer = writer.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row]);
}

#[test]
fn roundtrip_datetime() {
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
    #[derive(npyz::Serialize, npyz::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        datetime: i64,
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

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().dtype(dtype.clone()).writer(&mut writer).begin_1d().unwrap();
    out_file.push(&row).unwrap();
    out_file.finish().unwrap();

    let buffer = writer.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row]);
}

#[test]
fn roundtrip_bytes() {
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
    #[derive(npyz::Serialize, npyz::Deserialize)]
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

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().dtype(dtype.clone()).writer(&mut writer).begin_1d().unwrap();
    out_file.push(&row).unwrap();
    out_file.finish().unwrap();

    let buffer = writer.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row]);
}

// check that all byte orders are identical for bytestrings
// (i.e. don't accidentally reverse the bytestrings)
#[test]
fn roundtrip_bytes_byteorder() {
    #[derive(npyz::Serialize, npyz::Deserialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        s_le: Vec<u8>,
        s_be: Vec<u8>,
        s_na: Vec<u8>,
        v_le: Vec<u8>,
        v_be: Vec<u8>,
        v_na: Vec<u8>,
    }

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

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().dtype(dtype.clone()).writer(&mut writer).begin_1d().unwrap();
    out_file.push(&row).unwrap();
    out_file.finish().unwrap();

    let buffer = writer.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row]);
}

#[test]
fn nested_array_of_struct() {
    #[derive(npyz::Deserialize, npyz::Serialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq, Clone, Copy, Default)]
    struct Outer {
        foo: [Inner; 3],
    }

    #[derive(npyz::Deserialize, npyz::Serialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq, Clone, Copy, Default)]
    struct Inner {
        bar: f64,
    }

    let dtype = DType::Record(vec![
        Field { name: "foo".into(), dtype: DType::Array(3, Box::new(DType::Record(vec![
            plain_field("bar", "<f8"),
        ])))},
    ]);

    let row = Outer {
        foo: [
            Inner { bar: 1.0 },
            Inner { bar: 2.0 },
            Inner { bar: 3.0 },
        ],
    };

    let expected_data_bytes = {
        let mut buf = vec![];
        for x in vec![1.0, 2.0, 3.0] {
            buf.extend_from_slice(&f64::to_bits(x).to_le_bytes());
        }
        buf
    };

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().dtype(dtype.clone()).writer(&mut writer).begin_1d().unwrap();
    out_file.push(&row).unwrap();
    out_file.finish().unwrap();

    let buffer = writer.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Outer>().unwrap(), vec![row]);
}

#[test]
fn roundtrip_zero_length_array_member() {
    // Similar to:
    //
    // ```
    // import numpy as np
    //
    // arr = np.array([
    //     (3, np.zeros((3, 0, 7))),
    //     (4, np.zeros((3, 0, 7))),
    // ], dtype=[
    //     ('a', '<i4'),
    //     ('b', '<i4', [3, 0, 7]),
    // ])
    // ```
    #[derive(npyz::Serialize, npyz::Deserialize)]
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Row {
        a: i32,
        b: [[[i32; 7]; 0]; 3],
    }

    let dtype = DType::Record(vec![
        Field { name: "a".into(), dtype: DType::Plain("<i4".parse().unwrap()) },
        Field { name: "b".into(), dtype: DType::Array(3, Box::new(
            DType::Array(0, Box::new(
                DType::Array(7, Box::new(
                    DType::Plain("<i4".parse().unwrap())
                )),
            )),
        ))},
    ]);

    let row_0 = Row { a: 3, b: [[], [], []] };
    let row_1 = Row { a: 4, b: [[], [], []] };

    let expected_data_bytes = {
        let mut buf = vec![];
        buf.extend_from_slice(&i32::to_le_bytes(3));
        buf.extend_from_slice(&i32::to_le_bytes(4));
        buf
    };

    let mut writer = io::Cursor::new(vec![]);
    let mut out_file = npyz::WriteOptions::new().dtype(dtype.clone()).writer(&mut writer).begin_1d().unwrap();
    out_file.extend(vec![row_0, row_1]).unwrap();
    out_file.finish().unwrap();

    let buffer = writer.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row_0, row_1]);
}

// Try ndim == 0
#[test]
fn roundtrip_scalar() {
    // This is format.npy in a bsr formatted matrix.
    type Row = i32;
    let row: Row = 1;
    let dtype = DType::new_scalar("<i4".parse().unwrap());

    let expected_data_bytes = b"\x01\x00\x00\x00".to_vec();

    let mut cursor = Cursor::new(vec![]);
    let mut writer = {
        npyz::WriteOptions::new()
            .dtype(dtype.clone())
            .shape(&[])
            .writer(&mut cursor)
            .begin_nd().unwrap()
    };
    writer.push(&row).unwrap();
    writer.finish().unwrap();

    let buffer = cursor.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row]);
}

// try a unicode field name, which forces version 3
#[test]
fn roundtrip_version3() {
    #[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq, Clone)]
    struct Row {
        num: i32,
        αβ: i32,
    }

    let dtype = DType::Record(vec![
        plain_field("num", "<i4"),
        plain_field("αβ", "<i4"),
    ]);

    let row = Row { num: 1, αβ: 2 };
    let expected_data_bytes = b"\x01\x00\x00\x00\x02\x00\x00\x00".to_vec();

    let mut cursor = Cursor::new(vec![]);
    let mut writer = {
        npyz::WriteOptions::new()
            .dtype(dtype.clone())
            .writer(&mut cursor)
            .shape(&[1])
            .begin_nd().unwrap()
    };
    writer.push(&row).unwrap();
    writer.finish().unwrap();

    let buffer = cursor.into_inner();
    assert!(buffer.ends_with(&expected_data_bytes));

    assert_version(&buffer, (3, 0));

    let data = npyz::NpyFile::new(&buffer[..]).unwrap();
    assert_eq!(data.dtype(), dtype);
    assert_eq!(data.into_vec::<Row>().unwrap(), vec![row]);
}

#[track_caller]
fn assert_version(npy_bytes: &[u8], expected: (u8, u8)) {
    assert_eq!(&npy_bytes[6..8], &[expected.0, expected.1]);
}
