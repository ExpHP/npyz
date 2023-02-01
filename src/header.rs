
use std::io;
use core::fmt;

use py_literal::ParseError;
pub use py_literal::Value;
use byteorder::{LittleEndian, ReadBytesExt};
use num_bigint::Sign;

use crate::type_str::TypeStr;

/// Representation of a Numpy type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    /// Numpy type string. First character is `'>'` for big endian, `'<'` for little endian,
    /// or can be `'|'` if it doesn't matter.
    ///
    /// Examples: `>i4`, `<u8`, `>f8`, `|S7`. The number usually corresponds to the number of
    /// bytes (with the single exception of unicode strings `|U3`).
    Plain(TypeStr),

    /// Fixed-size inner array type.
    ///
    /// This is only possible inside structured arrays, where fields can themselves be arrays.
    /// E.g. in the `DType` for `dtype=[('abc', 'i4', [2, 3])]`, the `DType` for `abc`
    /// will be `Array(2, Array(3, Plain("<i4")))`. In rust, such an array could be read using
    /// the following element type:
    #[cfg_attr(any(not(doctest), feature="derive"), doc = r##"
```
# #[allow(unused)]
#[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
struct Row {
    abc: [[i32; 3]; 2],
}
```
"##)]
    Array(u64, Box<DType>),

    /// A structure record array
    Record(Vec<Field>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A field of a structured array dtype
pub struct Field {
    /// The name of the field
    pub name: String,

    /// The dtype of the field
    pub dtype: DType,
}

impl DType {
    /// Numpy format description of record dtype.
    ///
    /// Calling `descr` on [`DType::Array`] will not produce a valid python expression
    /// (the string will only be suitable for error messages).
    pub fn descr(&self) -> String {
        use DType::*;
        match self {
            Plain(ty) => format!("'{}'", ty),
            Record(fields) => {
                fields.iter()
                    .map(|Field { name, dtype }| {
                        let name = PyUtf8StringLiteral(name);
                        match dtype {
                            ty@Plain(_) |
                            ty@Record(_) => format!("({}, {}), ", name, ty.descr()),

                            array@Array(..) => {
                                let (shape, elem_ty) = extract_full_array_shape(array);
                                let shape_str = shape.iter().fold(String::new(), |o, n| o + &format!("{},", n));
                                format!("({}, {}, ({})), ", name, elem_ty.descr(), shape_str)
                            },
                        }
                    }).fold("[".to_string(), |o, n| o + &n) + "]"
            },

            Array(n, inner) => format!("<< array {} of {} >>", n, inner.descr()),
        }
    }

    // Create from description AST
    pub(crate) fn from_descr(descr: &Value) -> io::Result<Self> {
        use DType::*;
        match descr {
            Value::String(string) => Ok(Self::new_scalar(convert_string_to_type_str(string)?)),
            Value::List(list) => Ok(Record(convert_list_to_record_fields(list)?)),
            _ => Err(invalid_data("must be string or list")),
        }
    }

    // not part of stable API, but needed by the serialize_array test
    #[doc(hidden)]
    pub fn parse(source: &str) -> io::Result<Self> {
        let descr = parse_header_text_to_io_result(source.as_bytes())?;
        Self::from_descr(&descr)
    }

    /// Construct a scalar `DType`. (one which is not a nested array or record type)
    pub fn new_scalar(ty: TypeStr) -> Self {
        DType::Plain(ty)
    }

    /// Return a `TypeStr` only if the `DType` is a primitive scalar. (no arrays or record types)
    pub(crate) fn as_scalar(&self) -> Option<&TypeStr> {
        match self {
            DType::Plain(ty) => Some(ty),
            _ => None,
        }
    }

    /// Get the number of bytes that each item of this type occupies.
    ///
    /// If this value overflows the plaform's `usize` datatype, returns `None`.
    pub fn num_bytes(&self) -> Option<usize> {
        match self {
            DType::Plain(ty) => ty.num_bytes(),
            DType::Array(n, inner) => inner.num_bytes()?.checked_mul(usize::try_from(*n).ok()?),
            DType::Record(fields) => {
                fields.iter().map(|field| field.dtype.num_bytes())
                    .fold(Some(0), |a, b| a?.checked_add(b?))
            },
        }
    }
}

fn convert_list_to_record_fields(values: &[Value]) -> io::Result<Vec<Field>> {
    values.iter()
        .map(|value| match *value {
            Value::List(ref tuple) => convert_tuple_to_record_field(tuple),
            Value::Tuple(ref tuple) => convert_tuple_to_record_field(tuple),
            _ => Err(invalid_data("list must contain list or tuple"))
        })
        .collect()
}

fn convert_tuple_to_record_field(tuple: &[Value]) -> io::Result<Field> {
    match tuple.len() {
        2 | 3 => {}
        _ => return Err(invalid_data("list entry must contain 2 or 3 items")),
    }
    let name = match &tuple[0] {
        Value::String(name) => name.clone(),
        _ => return Err(invalid_data("list entry must contain a string for id")),
    };

    let mut dtype = DType::from_descr(&tuple[1])?;
    if let Some(s) = tuple.get(2) {
        let mut shape = convert_value_to_shape(s)?;
        while let Some(dim) = shape.pop() {
            dtype = DType::Array(dim, Box::new(dtype));
        }
    };

    Ok(Field { name, dtype })
}

fn convert_value_to_sequence(field: &Value) -> Option<&[Value]> {
    match field {
        &Value::List(ref lengths) => Some(lengths),
        &Value::Tuple(ref lengths) => Some(lengths),
        _ => None
    }
}

fn convert_value_to_shape_integer(number: &Value) -> io::Result<u64> {
    if let Value::Integer(number) = number {
        let parts = number.to_u64_digits();
        match parts {
            (Sign::Minus, _) => Err(invalid_data("dimension cannot be negative")),
            (Sign::NoSign, _) => Ok(0),
            (_, parts) if parts.len() == 1 => Ok(parts[0]),
            _ => Err(invalid_data("dimension cannot be larger than u64"))
        }
    } else {
        Err(invalid_data("dimension must be an integer"))
    }
}

fn extract_full_array_shape(mut dtype: &DType) -> (Vec<u64>, &DType) {
    let mut shape = vec![];
    while let &DType::Array(dim, ref inner) = dtype {
        shape.push(dim);
        dtype = inner;
    }
    (shape, dtype)
}

pub(crate) fn convert_value_to_shape(field: &Value) -> io::Result<Vec<u64>> {
    convert_value_to_sequence(field)
        .map(|f| f.iter().map(convert_value_to_shape_integer).collect())
        .ok_or(invalid_data("shape must be list or tuple"))?
}

fn convert_string_to_type_str(string: &str) -> io::Result<TypeStr> {
    match string.parse() {
        Ok(ty) => Ok(ty),
        Err(e) => Err(invalid_data(format_args!("invalid type string: {}", e))),
    }
}

fn invalid_data(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

pub(crate) fn read_header(r: &mut dyn io::Read) -> io::Result<Value> {
    let PreHeader { version_props, header_size } = read_pre_header(r)?;

    // FIXME: properly account for encoding
    let _ = version_props.encoding;
    let mut header_text = vec![0; header_size];
    r.read_exact(&mut header_text)?;

    parse_header_text_to_io_result(&header_text)
}

fn parse_header_text_to_io_result(bytes: &[u8]) -> io::Result<Value> {
    let without_newline = match bytes.split_last() {
        Some((&b'\n', rest)) => rest,
        _ => bytes,
    };
    std::str::from_utf8(without_newline)
        .map_err(|_| invalid_data("could not parse utf-8"))?
        .parse()
        .map_err(|e: ParseError| invalid_data(format_args!("could not parse Python expression: {}", e.to_string())))
}

struct PreHeader {
    version_props: VersionProps,
    header_size: usize,
}

fn read_pre_header(r: &mut dyn io::Read) -> io::Result<PreHeader> {
    let version = read_magic_and_version(r)?;
    let version_props = get_version_props(version)?;

    let header_size = match version_props.header_size_type {
        HeaderSizeType::U32 => r.read_u32::<LittleEndian>()? as usize,
        HeaderSizeType::U16 => r.read_u16::<LittleEndian>()? as usize,
    };

    Ok(PreHeader { version_props, header_size })
}

fn read_magic_and_version(r: &mut dyn io::Read) -> io::Result<(u8, u8)> {
    let magic_err = || invalid_data("magic not found for NPY file");

    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => magic_err(),
        _ => e,
    })?;

    if &buf[..6] != b"\x93NUMPY" {
        return Err(magic_err());
    }
    Ok((buf[6], buf[7]))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderSizeType { U16, U32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderEncoding {
    // Note: there is a suspicious phrase in the documentation:
    //
    //    "replaces the ASCII string (which in practice was latin1)"
    //
    // ...which suggests that this might actually be ANSI and not ASCII?
    Ascii,
    Utf8,
}
#[derive(Debug, Clone)]
pub(crate) struct VersionProps {
    pub(crate) header_size_type: HeaderSizeType,
    pub(crate) encoding: HeaderEncoding,
}
impl VersionProps {
    pub fn bytes_before_text(&self) -> usize {
        match self.header_size_type {
            HeaderSizeType::U16 => 10,
            HeaderSizeType::U32 => 12,
        }
    }
}

pub(crate) fn get_version_props(version: (u8, u8)) -> io::Result<VersionProps> {
    use self::HeaderSizeType::*;
    use self::HeaderEncoding::*;
    match version {
        (1, 0) => Ok(VersionProps { header_size_type: U16, encoding: Ascii }),
        (2, 0) => Ok(VersionProps { header_size_type: U32, encoding: Ascii }),
        (3, 0) => Ok(VersionProps { header_size_type: U32, encoding: Utf8 }),
        _ => Err(invalid_data(format_args!("unsupported version: ({}, {})", version.0, version.1))),
    }
}

/// Formats a python string literal.
///
/// Unlike the [`Display`] impl for [`py_literal`], the string is encoded in
/// UTF-8 (supported by NPY version 3), resulting in fewer escapes.
struct PyUtf8StringLiteral<'a>(&'a str);

impl fmt::Display for PyUtf8StringLiteral<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let replaced = {
            self.0.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\r", "\\r")
                .replace("\n", "\\n")
        };
        write!(f, "'{}'", replaced)
    }
}

pub(crate) fn get_minimal_version(required_props: VersionProps) -> (u8, u8) {
    if required_props.encoding == HeaderEncoding::Utf8 {
        (3, 0)
    } else if required_props.header_size_type == HeaderSizeType::U32 {
        (2, 0)
    } else {
        (1, 0)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    type TestResult = std::result::Result<(), Box<dyn Error>>;

    #[test]
    fn description_of_record_array_as_python_list_of_tuples() -> TestResult {
        let dtype = DType::Record(vec![
            Field {
                name: "float".to_string(),
                dtype: DType::Plain(">f4".parse()?),
            },
            Field {
                name: "byte".to_string(),
                dtype: DType::Plain("<u1".parse()?),
            }
        ]);
        let expected = "[('float', '>f4'), ('byte', '<u1'), ]";
        assert_eq!(dtype.descr(), expected);
        Ok(())
    }

    #[test]
    fn description_of_unstructured_primitive_array() -> TestResult {
        let dtype = DType::Plain(">f8".parse()?);
        assert_eq!(dtype.descr(), "'>f8'");
        Ok(())
    }

    #[test]
    fn description_of_nested_record_dtype() -> TestResult {
        let dtype = DType::Record(vec![
            Field {
                name: "parent".to_string(),
                dtype: DType::Record(vec![
                    Field {
                        name: "child".to_string(),
                        dtype: DType::Plain("<i4".parse()?),
                    },
                ]),
            }
        ]);
        assert_eq!(dtype.descr(), "[('parent', [('child', '<i4'), ]), ]");
        Ok(())
    }

    #[test]
    fn converts_simple_description_to_record_dtype() -> TestResult {
        let dtype = ">f8";
        assert_eq!(
            DType::from_descr(&Value::String(dtype.to_string())).unwrap(),
            DType::Plain(dtype.parse()?),
        );
        Ok(())
    }

    #[test]
    fn converts_non_endian_description_to_record_dtype() -> TestResult {
        let dtype = "|u1";
        assert_eq!(
            DType::from_descr(&Value::String(dtype.to_string())).unwrap(),
            DType::Plain(dtype.parse()?),
        );
        Ok(())
    }

    #[test]
    fn converts_record_description_to_record_dtype() -> TestResult {
        let descr = parse("[('a', '<u2'), ('b', '<f4')]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "a".to_string(),
                dtype: DType::Plain("<u2".parse()?),
            },
            Field {
                name: "b".to_string(),
                dtype: DType::Plain("<f4".parse()?),
            }
        ]);
        assert_eq!(DType::from_descr(&descr).unwrap(), expected_dtype);
        Ok(())
    }

    #[test]
    fn funny_member_name_roundtrips() -> TestResult {
        let original_dtype = DType::Record(vec![
            Field {
                name: " \'\"\r\n\\ ".to_string(),
                dtype: DType::Plain("<u2".parse()?),
            },
        ]);
        let descr = parse(&original_dtype.descr());
        assert_eq!(DType::from_descr(&descr).unwrap(), original_dtype);
        Ok(())
    }

    #[test]
    fn record_description_with_onedimensional_field_shape_declaration() -> TestResult {
        let descr = parse("[('a', '>f8', (1,))]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "a".to_string(),
                dtype: DType::Array(1, Box::new(DType::Plain(">f8".parse()?))),
            }
        ]);
        assert_eq!(DType::from_descr(&descr).unwrap(), expected_dtype);
        Ok(())
    }

    #[test]
    fn record_description_with_nested_record_field() -> TestResult {
        let descr = parse("[('parent', [('child', '<i4')])]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "parent".to_string(),
                dtype: DType::Record(vec![
                    Field {
                        name: "child".to_string(),
                        dtype: DType::Plain("<i4".parse()?),
                    },
                ]),
            }
        ]);
        assert_eq!(DType::from_descr(&descr).unwrap(), expected_dtype);
        Ok(())
    }

    #[test]
    fn nested_record_field_array() -> TestResult {
        let descr = parse("[('parent', [('child', '<i4')], (2,))]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "parent".to_string(),
                dtype: DType::Array(2, Box::new(DType::Record(vec![
                    Field {
                        name: "child".to_string(),
                        dtype: DType::Plain("<i4".parse()?),
                    },
                ]))),
            }
        ]);
        assert_eq!(DType::from_descr(&descr).unwrap(), expected_dtype);
        Ok(())
    }

    #[test]
    fn errors_on_value_variants_that_cannot_be_converted() {
        let no_dtype = Value::Boolean(false);
        assert!(DType::from_descr(&no_dtype).is_err());
    }

    #[test]
    fn errors_when_record_list_does_not_contain_lists() {
        let faulty_list = parse("['a', 123]");
        assert!(DType::from_descr(&faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_few_items() {
        let faulty_list = parse("[('a',)]");
        assert!(DType::from_descr(&faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_many_items() {
        let faulty_list = parse("[('a', 1, 2, 3)]");
        assert!(DType::from_descr(&faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_non_strings_for_id_or_dtype() {
        let faulty_list = parse("[(1, 2)]");
        assert!(DType::from_descr(&faulty_list).is_err());
    }

    #[test]
    fn errors_when_shape_is_not_a_list() {
        let no_shape = parse("1");
        assert!(convert_value_to_shape(&no_shape).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_a_number() {
        let no_number = parse("[]");
        assert!(convert_value_to_shape_integer(&no_number).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_negative() {
        assert!(convert_value_to_shape_integer(&parse("-1")).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_larger_than_u64() {
        assert!(convert_value_to_shape_integer(&parse("18446744073709551616")).is_err());
    }

    fn parse(source: &str) -> Value {
        source.parse().unwrap_or_else(|e| panic!("could not parse Python expression:\n{}", e))
    }
}
