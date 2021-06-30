
use nom::IResult;
use byteorder::{LE, ReadBytesExt};
use std::collections::HashMap;
use std::io;
use type_str::TypeStr;

/// Representation of a Numpy type
#[derive(PartialEq, Eq, Debug, Clone)]
pub enum DType {
    /// A simple array with only a single field
    Plain {
        /// Numpy type string. First character is `'>'` for big endian, `'<'` for little endian,
        /// or can be `'|'` if it doesn't matter.
        ///
        /// Examples: `>i4`, `<u8`, `>f8`, `|S7`. The number usually corresponds to the number of
        /// bytes (with the single exception of unicode strings `|U3`).
        ty: TypeStr,

        /// Shape of a type.
        ///
        /// Scalar has zero entries. Otherwise, number of entries == number of dimensions and each
        /// entry specifies size in the respective dimension.
        shape: Vec<u64>
    },

    /// A structure record array
    Record(Vec<Field>)
}

#[derive(PartialEq, Eq, Debug, Clone)]
/// A field of a record dtype
pub struct Field {
    /// The name of the field
    pub name: String,

    /// The dtype of the field
    pub dtype: DType
}

impl DType {
    /// Numpy format description of record dtype.
    pub fn descr(&self) -> String {
        use DType::*;
        match *self {
            Record(ref fields) =>
                fields.iter()
                    .map(|&Field { ref name, ref dtype }|
                        match *dtype {
                            Plain { ref ty, ref shape } =>
                                if shape.len() == 0 {
                                    format!("('{}', '{}'), ", name, ty)
                                } else {
                                    let shape_str = shape.iter().fold(String::new(), |o, n| o + &format!("{},", n));
                                    format!("('{}', '{}', ({})), ", name, ty, shape_str)
                                },
                            ref record@Record(_) => {
                                format!("('{}', {}), ", name, record.descr())
                            },
                        }
                    )
                    .fold("[".to_string(), |o, n| o + &n) + "]",
            Plain { ref ty, .. } => format!("'{}'", ty),
        }
    }

    /// Create from description AST
    pub fn from_descr(descr: Value) -> io::Result<Self> {
        use DType::*;
        match descr {
            Value::String(ref string) => Ok(Self::new_scalar(convert_string_to_type_str(string)?)),
            Value::List(ref list) => Ok(Record(convert_list_to_record_fields(list)?)),
            _ => Err(invalid_data("must be string or list")),
        }
    }

    // not part of stable API, but needed by the serialize_array test
    #[doc(hidden)]
    pub fn parse(source: &str) -> io::Result<Self> {
        let descr = parse_header_text_to_io_result(source.as_bytes())?;
        Self::from_descr(descr)
    }

    /// Construct a scalar `DType`. (one which is not a nested array or record type)
    pub fn new_scalar(ty: TypeStr) -> Self {
        DType::Plain { ty, shape: vec![] }
    }

    /// Return a `TypeStr` only if the `DType` is a primitive scalar. (no arrays or record types)
    pub(crate) fn as_scalar(&self) -> Option<&TypeStr> {
        match self {
            DType::Plain { ty, shape } if shape.is_empty() => Some(ty),
            _ => None,
        }
    }

    /// Get the number of bytes that each item of this type occupies.
    pub fn num_bytes(&self) -> usize {
        match self {
            DType::Plain { ty, shape } => {
                ty.num_bytes() * shape.iter().product::<u64>() as usize
            },
            DType::Record(fields) => {
                fields.iter().map(|field| field.dtype.num_bytes()).sum()
            },
        }
    }
}

fn convert_list_to_record_fields(values: &[Value]) -> io::Result<Vec<Field>> {
    values.iter()
        .map(|value| match *value {
            Value::List(ref tuple) => convert_tuple_to_record_field(tuple),
            _ => Err(invalid_data("list must contain list or tuple"))
        })
        .collect()
}

fn convert_tuple_to_record_field(tuple: &[Value]) -> io::Result<Field> {
    use self::Value::{String, List};

    match tuple.len() {
        2 | 3 => match (&tuple[0], &tuple[1], tuple.get(2)) {
            (&String(ref name), &String(ref dtype), ref shape) =>
                Ok(Field { name: name.clone(), dtype: DType::Plain {
                    ty: convert_string_to_type_str(dtype)?,
                    shape: if let &Some(ref s) = shape {
                        convert_value_to_field_shape(s)?
                    } else {
                        vec![]
                    }
                } }),
            (&String(ref name), &List(ref list), None) =>
                Ok(Field {
                    name: name.clone(),
                    dtype: DType::Record(convert_list_to_record_fields(list)?)
                }),
            (&String(_), &List(_), Some(_)) =>
                Err(invalid_data("nested arrays of Record types are not supported.")),
            _ =>
                Err(invalid_data("list entry must contain a string for id and a valid dtype")),
        },
        _ => Err(invalid_data("list entry must contain 2 or 3 items")),
    }
}

// FIXME: Remove; no reason to forbid size 0
fn convert_value_to_field_shape(field: &Value) -> io::Result<Vec<u64>> {
    if let Value::List(ref lengths) = *field {
        lengths.iter().map(convert_value_to_positive_integer).collect()
    } else {
        Err(invalid_data("shape must be list or tuple"))
    }
}

fn convert_value_to_positive_integer(number: &Value) -> io::Result<u64> {
    if let Value::Integer(number) = *number {
        if number > 0 {
            Ok(number as u64)
        } else {
            Err(invalid_data("number must be positive"))
        }
    } else {
        Err(invalid_data("must be a number"))
    }
}

pub(crate) fn convert_value_to_shape(field: &Value) -> io::Result<Vec<u64>> {
    if let Value::List(ref lengths) = *field {
        lengths.iter().map(convert_value_to_shape_integer).collect()
    } else {
        Err(invalid_data("shape must be list or tuple"))
    }
}

pub fn convert_value_to_shape_integer(number: &Value) -> io::Result<u64> {
    if let Value::Integer(number) = *number {
        if number >= 0 {
            Ok(number as u64)
        } else {
            Err(invalid_data("shape integer cannot be negative"))
        }
    } else {
        Err(invalid_data("shape elements must be number"))
    }
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

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Value {
    String(String),
    Integer(i64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
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
    match parser::value(bytes) {
        IResult::Done(remainder, header) => {
            if !remainder.iter().all(|b| b" \t\n\r\0".contains(b)) {
                return Err(invalid_data(format_args!("unexpected trailing data in header: {:?}", remainder)));
            }
            Ok(header)
        },
        IResult::Incomplete(needed) => {
            Err(invalid_data(format_args!("could not parse Python expression: {:?}", needed)))
        },
        IResult::Error(err) => {
            Err(invalid_data(format_args!("could not parse Python expression: {:?}", err)))
        },
    }
}

struct PreHeader {
    version_props: VersionProps,
    header_size: usize,
}

fn read_pre_header(r: &mut dyn io::Read) -> io::Result<PreHeader> {
    let version = read_magic_and_version(r)?;
    let version_props = get_version_props(version)?;

    let header_size = match version_props.header_size_type {
        HeaderSizeType::U32 => r.read_u32::<LE>()? as usize,
        HeaderSizeType::U16 => r.read_u16::<LE>()? as usize,
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum HeaderSizeType { U16, U32 }

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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

pub(crate) fn get_minimal_version(required_props: VersionProps) -> (u8, u8) {
    if required_props.encoding == HeaderEncoding::Utf8 {
        (3, 0)
    } else if required_props.header_size_type == HeaderSizeType::U32 {
        (2, 0)
    } else {
        (1, 0)
    }
}

mod parser {
    use super::Value;
    use nom::*;

    named!(pub value<Value>, alt!(integer | boolean | string | list | map));

    named!(pub integer<Value>,
        map!(
            map_res!(
                map_res!(
                    ws!(digit),
                    ::std::str::from_utf8
                ),
                ::std::str::FromStr::from_str
            ),
            Value::Integer
        )
    );

    named!(pub boolean<Value>,
        ws!(alt!(
            tag!("True") => { |_| Value::Bool(true) } |
            tag!("False") => { |_| Value::Bool(false) }
        ))
    );

    named!(pub string<Value>,
        map!(
            map!(
                map_res!(
                    ws!(alt!(
                        delimited!(tag!("\""),
                            is_not_s!("\""),
                            tag!("\"")) |
                        delimited!(tag!("\'"),
                            is_not_s!("\'"),
                            tag!("\'"))
                        )),
                    ::std::str::from_utf8
                ),
                |s: &str| s.to_string()
            ),
            Value::String
        )
    );

    named!(pub list<Value>,
        map!(
            ws!(alt!(
                delimited!(tag!("["),
                    terminated!(separated_list!(tag!(","), value), alt!(tag!(",") | tag!(""))),
                    tag!("]")) |
                delimited!(tag!("("),
                    terminated!(separated_list!(tag!(","), value), alt!(tag!(",") | tag!(""))),
                    tag!(")"))
            )),
            Value::List
        )
    );

    named!(pub map<Value>,
        map!(
            ws!(
                delimited!(tag!("{"),
                    terminated!(separated_list!(tag!(","),
                        separated_pair!(map_opt!(string, |it| match it { Value::String(s) => Some(s), _ => None }), tag!(":"), value)
                    ), alt!(tag!(",") | tag!(""))),
                    tag!("}"))
            ),
            |v: Vec<_>| Value::Map(v.into_iter().collect())
        )
    );
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
                dtype: DType::Plain { ty: ">f4".parse()?, shape: vec![] }
            },
            Field {
                name: "byte".to_string(),
                dtype: DType::Plain { ty: "<u1".parse()?, shape: vec![] }
            }
        ]);
        let expected = "[('float', '>f4'), ('byte', '<u1'), ]";
        assert_eq!(dtype.descr(), expected);
        Ok(())
    }

    #[test]
    fn description_of_unstructured_primitive_array() -> TestResult {
        let dtype = DType::Plain { ty: ">f8".parse()?, shape: vec![] };
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
                        dtype: DType::Plain { ty: "<i4".parse()?, shape: vec![] }
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
            DType::from_descr(Value::String(dtype.to_string())).unwrap(),
            DType::Plain { ty: dtype.parse()?, shape: vec![] }
        );
        Ok(())
    }

    #[test]
    fn converts_non_endian_description_to_record_dtype() -> TestResult {
        let dtype = "|u1";
        assert_eq!(
            DType::from_descr(Value::String(dtype.to_string())).unwrap(),
            DType::Plain { ty: dtype.parse()?, shape: vec![] }
        );
        Ok(())
    }

    #[test]
    fn converts_record_description_to_record_dtype() -> TestResult {
        let descr = parse("[('a', '<u2'), ('b', '<f4')]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "a".to_string(),
                dtype: DType::Plain { ty: "<u2".parse()?, shape: vec![] }
            },
            Field {
                name: "b".to_string(),
                dtype: DType::Plain { ty: "<f4".parse()?, shape: vec![] }
            }
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
        Ok(())
    }

    #[test]
    fn record_description_with_onedimensional_field_shape_declaration() -> TestResult {
        let descr = parse("[('a', '>f8', (1,))]");
        let expected_dtype = DType::Record(vec![
            Field {
                name: "a".to_string(),
                dtype: DType::Plain { ty: ">f8".parse()?, shape: vec![1] }
            }
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
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
                        dtype: DType::Plain { ty: "<i4".parse()?, shape: vec![] }
                    },
                ]),
            }
        ]);
        assert_eq!(DType::from_descr(descr).unwrap(), expected_dtype);
        Ok(())
    }


    #[test]
    fn errors_on_nested_record_field_array() {
        let descr = parse("[('parent', [('child', '<i4')], (2,))]");
        assert!(DType::from_descr(descr).is_err());
    }

    #[test]
    fn errors_on_value_variants_that_cannot_be_converted() {
        let no_dtype = Value::Bool(false);
        assert!(DType::from_descr(no_dtype).is_err());
    }

    #[test]
    fn errors_when_record_list_does_not_contain_lists() {
        let faulty_list = parse("['a', 123]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_few_items() {
        let faulty_list = parse("[('a')]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_too_many_items() {
        let faulty_list = parse("[('a', 1, 2, 3)]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_record_list_entry_contains_non_strings_for_id_or_dtype() {
        let faulty_list = parse("[(1, 2)]");
        assert!(DType::from_descr(faulty_list).is_err());
    }

    #[test]
    fn errors_when_shape_is_not_a_list() {
        let no_shape = parse("1");
        assert!(convert_value_to_field_shape(&no_shape).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_a_number() {
        let no_number = parse("[]");
        assert!(convert_value_to_positive_integer(&no_number).is_err());
    }

    #[test]
    fn errors_when_shape_number_is_not_positive() {
        assert!(convert_value_to_positive_integer(&parse("0")).is_err());
    }

    fn parse(source: &str) -> Value {
        parser::value(source.as_bytes())
            .to_result()
            .expect("could not parse Python expression")
    }
}
