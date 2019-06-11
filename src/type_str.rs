use std::fmt;

/// Represents an Array Interface type-string.
///
/// This is more or less the `DType` of a scalar type.
/// Exposes a `FromStr` impl for construction, and a `Display` impl for writing.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// use npy::TypeStr;
///
/// let ts = "|i1".parse::<TypeStr>()?;
///
/// assert_eq!(format!("{}", ts), "|i1");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeStr {
    pub(crate) endianness: Endianness,
    pub(crate) type_kind: TypeKind,
    pub(crate) size: u64,
    pub(crate) time_units: Option<TimeUnits>,
}

/// Represents the first character in a type-string.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Endianness {
    /// Code `<`.
    Little,
    /// Code `>`.
    Big,
    /// Code `|`. Used when endianness is irrelevant.
    ///
    /// Only valid when the size is `1`, or when `kind` is `TypeKind::Other`
    /// or `TypeKind::ByteStr`.
    Irrelevant,
}

impl Endianness {
    fn from_char(s: char) -> Option<Self> {
        match s {
            '<' => Some(Endianness::Little),
            '>' => Some(Endianness::Big),
            '|' => Some(Endianness::Irrelevant),
            _ => None,
        }
    }

    fn to_str(self) -> &'static str {
        match self {
            Endianness::Little => "<",
            Endianness::Big => ">",
            Endianness::Irrelevant => "|",
        }
    }
}

impl Endianness {
    pub(crate) fn of_machine() -> Self {
        match i32::from_be(0x00_00_00_01) {
            0x00_00_00_01 => Endianness::Big,
            0x01_00_00_00 => Endianness::Little,
            _ => unreachable!(),
        }
    }

    /// Returns `true` if byteorder swapping is necessary between two types.
    pub(crate) fn requires_swap(self, other: Endianness) -> bool {
        match (self, other) {
            (Endianness::Little, Endianness::Big) |
            (Endianness::Big, Endianness::Little) => true,

            _ => false,
        }
    }
}

/// Represents the second character in a type-string.
///
/// Indicates the type of data stored.  Affects the interpretation of `size` and `endianness`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TypeKind {
    /// Code `b`.
    ///
    /// `size` must be 1, and legal values are `0x00` (`false`) or `0x01` (`true`).
    Bool,
    /// Code `i`.
    ///
    /// Notice that numpy does not support 128-bit integers.
    Int,
    /// Code `u`.
    ///
    /// Notice that numpy does not support 128-bit integers.
    Uint,
    /// Code `f`.
    ///
    /// Notice that numpy **does** support 128-bit floats.
    Float,
    /// Code `c`. Represents a complex number.
    ///
    /// The real part followed by the imaginary part, with `size` bytes total between the two of
    /// them. Each part has the specified endianness, but the real part always comes first.
    Complex,
    /// Code `m`. Represents a `numpy.timedelta64`.
    ///
    /// Can use `i64` for serialization. `size` must be 8.
    /// Check [`PlainDtype::time_units`] for the units.
    TimeDelta,
    /// Code `M`. Represents a `numpy.datetime64`.
    ///
    /// Can use `u64` for serialization. `size` must be 8.
    /// Check [`PlainDtype::time_units`] for the units.
    DateTime,
    /// Code `S`. Represents a Python 3 `bytes` (`str` in Python 2).
    ///
    /// Can use `Vec<u8>` for serialization.
    ///
    /// A `bytes` of length `size`.  Strings shorter than this length are zero-padded on the right.
    /// This implies that they cannot contain trailing `NUL`s. (They can, however, contain interior
    /// `NUL`s).  To preserve trailing `NUL`s, use `RawData` (`V`) instead.
    ByteStr,
    /// Code `U`. Represents a Python 3 `str` (`unicode` in Python 2).
    ///
    /// A `str` that contains `size` code points (**not bytes!**). Each code unit is encoded as a
    /// 32-bit integer of the given endianness. Strings with fewer than `size` code units are
    /// zero-padded on the right. (thus they cannot contain trailing copies of U+0000 'NULL';
    /// they can, however, contain interior copies)
    ///
    /// Like Rust's `char`, the code points must have a value in `[0, 0x110000)`.  However, unlike
    /// `char`, surrogate code points are allowed.
    UnicodeStr,
    /// Code `V`.  Represents a binary blob of `size` bytes.
    ///
    /// Can use `Vec<u8>` for serialization.
    RawData,
}

impl TypeKind {
    fn from_char(s: char) -> Option<Self> {
        match s {
            'b' => Some(TypeKind::Bool),
            'i' => Some(TypeKind::Int),
            'u' => Some(TypeKind::Uint),
            'f' => Some(TypeKind::Float),
            'c' => Some(TypeKind::Complex),
            'm' => Some(TypeKind::TimeDelta),
            'M' => Some(TypeKind::DateTime),
            'S' => Some(TypeKind::ByteStr),
            'U' => Some(TypeKind::UnicodeStr),
            'V' => Some(TypeKind::RawData),
            _ => None,
        }
    }

    fn to_str(self) -> &'static str {
        match self {
            TypeKind::Bool => "b",
            TypeKind::Int => "i",
            TypeKind::Uint => "u",
            TypeKind::Float => "f",
            TypeKind::Complex => "c",
            TypeKind::TimeDelta => "m",
            TypeKind::DateTime => "M",
            TypeKind::ByteStr => "S",
            TypeKind::UnicodeStr => "U",
            TypeKind::RawData => "V",
        }
    }
}

impl TypeKind {
    // `None` means all sizes are valid.
    fn valid_sizes(self) -> Option<&'static [u64]> {
        match self {
            TypeKind::Bool => Some(&[1]),

            // numpy doesn't actually support 128-bit ints
            TypeKind::Int |
            TypeKind::Uint => Some(&[1, 2, 4, 8]),

            // yes, 128-bit floats are supported by numpy
            TypeKind::Float => Some(&[2, 4, 8, 16]),

            // 4-byte complex numbers are mysteriously missing from numpy
            TypeKind::Complex => Some(&[8, 16, 32]),

            TypeKind::TimeDelta |
            TypeKind::DateTime => Some(&[8]),

            // (Note: numpy does support types `|S0` and `|U0`, though for some reason `numpy.save`
            //        changes them to `|S1` and `|U1`.)
            TypeKind::ByteStr |
            TypeKind::UnicodeStr |
            TypeKind::RawData => None,
        }
    }

    /// Returns `true` if `|` endianness is illegal.
    fn requires_endianness(self, size: u64) -> bool {
        match self {
            TypeKind::Bool |
            TypeKind::Int |
            TypeKind::Uint |
            TypeKind::Float |
            TypeKind::TimeDelta |
            TypeKind::DateTime |
            TypeKind::Complex => size != 1,

            TypeKind::UnicodeStr => true,

            TypeKind::ByteStr |
            TypeKind::RawData => false,
        }
    }

    /// Returns `true` if `|` endianness is illegal.
    fn has_units(self) -> bool {
        match self {
            TypeKind::TimeDelta |
            TypeKind::DateTime => true,

            _ => false,
        }
    }
}

impl TypeStr {
    pub(crate) fn with_auto_endianness(type_kind: TypeKind, size: u64, time_units: Option<TimeUnits>) -> Self {
        let endianness = match type_kind.requires_endianness(size) {
            true => Endianness::of_machine(),
            false => Endianness::Irrelevant,
        };
        TypeStr { endianness, type_kind, size, time_units }.validate().unwrap()
    }

    /// The number of bytes for a single scalar value.
    pub(crate) fn num_bytes(&self) -> usize {
        match self.type_kind {
            TypeKind::Bool |
            TypeKind::Int |
            TypeKind::Uint |
            TypeKind::Float |
            TypeKind::Complex |
            TypeKind::TimeDelta |
            TypeKind::DateTime |
            TypeKind::ByteStr |
            TypeKind::RawData => self.size as usize,

            TypeKind::UnicodeStr => self.size as usize * 4,
        }
    }
}

/// Represents the units of the `m` and `M` datatypes.
///
/// These appear inside square brackets at the end of the `descr` string for these datatypes.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum TimeUnits {
    /// Code `Y`.
    Year,
    /// Code `M`.
    Month,
    /// Code `W`.
    Week,
    /// Code `D`.
    Day,
    /// Code `h`.
    Hour,
    /// Code `m`.
    Minute,
    /// Code `s`.
    Second,
    /// Code `ms`.
    Millisecond,
    /// Code `us`.
    Microsecond,
    /// Code `ns`.
    Nanosecond,
    /// Code `ps`.
    Picosecond,
    /// Code `fs`.
    Femtosecond,
    /// Code `as`.
    Attosecond,
}

impl TimeUnits {
    fn from_str(s: &str) -> Option<TimeUnits> {
        match s {
            "Y" => Some(TimeUnits::Year),
            "M" => Some(TimeUnits::Month),
            "W" => Some(TimeUnits::Week),
            "D" => Some(TimeUnits::Day),
            "h" => Some(TimeUnits::Hour),
            "m" => Some(TimeUnits::Minute),
            "s" => Some(TimeUnits::Second),
            "ms" => Some(TimeUnits::Millisecond),
            "us" => Some(TimeUnits::Microsecond),
            "ns" => Some(TimeUnits::Nanosecond),
            "ps" => Some(TimeUnits::Picosecond),
            "fs" => Some(TimeUnits::Femtosecond),
            "as" => Some(TimeUnits::Attosecond),
            _ => None,
        }
    }

    fn to_str(self) -> &'static str {
        match self {
            TimeUnits::Year => "Y",
            TimeUnits::Month => "M",
            TimeUnits::Week => "W",
            TimeUnits::Day => "D",
            TimeUnits::Hour => "h",
            TimeUnits::Minute => "m",
            TimeUnits::Second => "s",
            TimeUnits::Millisecond => "ms",
            TimeUnits::Microsecond => "us",
            TimeUnits::Nanosecond => "ns",
            TimeUnits::Picosecond => "ps",
            TimeUnits::Femtosecond => "fs",
            TimeUnits::Attosecond => "as",
        }
    }
}

impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.to_str(), f)
    }
}

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.to_str(), f)
    }
}

impl fmt::Display for TimeUnits {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.to_str(), f)
    }
}

impl fmt::Display for TypeStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}{}", self.endianness, self.type_kind, self.size)?;
        if let Some(time_units) = self.time_units {
            write!(f, "[{}]", time_units)?;
        }
        Ok(())
    }
}

pub use self::parse::ParseTypeStrError;
mod parse {
    use super::*;

    /// Error type returned by `<TypeStr as FromStr>::parse`.
    #[derive(Debug, Clone)]
    pub struct ParseTypeStrError(ErrorKind);

    #[derive(Debug, Clone)]
    enum ErrorKind {
        SyntaxError,
        ParseIntError(std::num::ParseIntError),
        InvalidEndianness(TypeStr),
        InvalidSize(TypeStr),
        MissingOrUnexpectedUnits(TypeStr),
    }

    impl fmt::Display for ParseTypeStrError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            use self::ErrorKind::*;

            match &self.0 {
                SyntaxError => write!(f, "Invalid type-string"),
                InvalidEndianness(ty) => write!(f, "Type string '{}' has invalid endianness", ty),
                InvalidSize(ty) => {
                    write!(f, "Type string '{}' has invalid size.", ty)?;
                    write!(f, " Valid sizes are: {:?}", ty.type_kind.valid_sizes().unwrap())?;
                    Ok(())
                },
                MissingOrUnexpectedUnits(ty) => {
                    if ty.type_kind.has_units() {
                        write!(f, "Type string '{}' is missing time units.", ty)
                    } else {
                        write!(f, "Unexpected time units in type string '{}'.", ty)
                    }
                },
                ParseIntError(e) => write!(f, "{}", e),
            }
        }
    }

    macro_rules! bail {
        ($variant:expr) => {
            return Err(ParseTypeStrError($variant))
        };
    }

    impl std::error::Error for ParseTypeStrError {}

    impl std::str::FromStr for TypeStr {
        type Err = ParseTypeStrError;

        fn from_str(input: &str) -> Result<Self, ParseTypeStrError> {
            use self::ErrorKind::*;

            if input.len() < 3 {
                bail!(SyntaxError);
            }

            let mut chars = input.chars();

            let c = chars.next().unwrap();
            let endianness = match Endianness::from_char(c) {
                None => bail!(SyntaxError),
                Some(v) => v,
            };

            let c = chars.next().unwrap();
            let type_kind = match TypeKind::from_char(c) {
                None => bail!(SyntaxError),
                Some(v) => v,
            };

            let remainder = chars.as_str();
            let size_end = {
                remainder.bytes().position(|b| !b.is_ascii_digit())
                    .unwrap_or(remainder.len())
            };
            if size_end == 0 {
                bail!(SyntaxError);
            }
            let (size, remainder) = remainder.split_at(size_end);
            let size = match size.parse() {
                Err(e) => bail!(ParseIntError(e)), // probably overflow
                Ok(v) => v,
            };

            let time_units = if remainder.is_empty() {
                None
            } else {
                let mut chars = remainder.chars();
                match (chars.next(), chars.next_back()) {
                    (Some('['), Some(']')) => {},
                    _ => bail!(SyntaxError),
                }

                match TimeUnits::from_str(chars.as_str()) {
                    None => bail!(SyntaxError),
                    Some(v) => Some(v),
                }
            };

            TypeStr { endianness, type_kind, size, time_units }
                .validate()
        }
    }

    impl TypeStr {
        pub(crate) fn validate(self) -> Result<Self, ParseTypeStrError> {
            use self::ErrorKind::*;

            let TypeStr { endianness, type_kind, size, time_units } = self;

            if type_kind.requires_endianness(size) && endianness == Endianness::Irrelevant {
                bail!(InvalidEndianness(self));
            }

            if let Some(valid_sizes) = type_kind.valid_sizes() {
                if !valid_sizes.contains(&size) {
                    bail!(InvalidSize(self));
                }
            }

            if type_kind.has_units() != time_units.is_some() {
                bail!(MissingOrUnexpectedUnits(self));
            }

            Ok(self)
        }
    }

    #[cfg(test)]
    #[deny(unused)]
    mod tests {
        use super::*;

        macro_rules! assert_matches {
            ($expr:expr, $pat:pat) => {
                match $expr {
                    $pat => {},
                    actual => panic!("Expected: {}\nGot: {:?}", stringify!($pat), actual),
                }
            };
        }

        macro_rules! check_ok {
            ($s:expr) => {
                assert_matches!($s.parse::<TypeStr>(), Ok(_));
            };
        }
        macro_rules! check_err {
            ($s:expr, $p:pat) => {
                assert_matches!($s.parse::<TypeStr>(), Err(ParseTypeStrError($p)));
            };
        }

        #[test]
        fn errors() {
            use self::ErrorKind::*;

            check_err!("", SyntaxError);
            check_err!(">", SyntaxError);
            check_err!(">m", SyntaxError);
            check_err!(">m8[", SyntaxError);
            check_err!(">m8[us", SyntaxError);
            check_ok!(">m8[us]");
            check_ok!(">m8[D]");
            check_err!(">m8[us]garbage", SyntaxError);
            check_err!(">m8[us]]", SyntaxError);


            check_err!("", SyntaxError);
            check_err!(">", SyntaxError);
            check_err!(">i", SyntaxError);
            check_ok!(">i8");
            check_ok!(">c16");
            check_err!(">i8garbage", SyntaxError);

            // length-zero integer
            check_err!(">m[us]", SyntaxError);
            check_err!(">i", SyntaxError);

            // make sure integer overflow doesn't panic
            check_err!(">m999999999999999999999999999999[us]", _);
            check_err!(">i999999999999999999999999999999", _);

            // Unrecognized specifiers
            check_ok!("<i8");
            check_err!("*i8", _);
            check_err!("<p8", _);
            check_ok!(">m8[us]");
            check_err!(">m8[bus]", _);
            check_err!(">m8[usb]", _);
            check_err!(">m8[xq]", _);

            // Required endianness
            check_ok!("|i1");
            check_ok!("|S7");
            check_ok!("|V7");
            check_err!("|i8", InvalidEndianness { .. });
            check_err!("|U1", InvalidEndianness { .. });

            // Size
            check_ok!(">i8");
            check_err!(">i9", InvalidSize { .. });
            check_err!(">m4[us]", InvalidSize { .. });
            check_err!(">b4", InvalidSize { .. });
            check_ok!("|S0");
            check_ok!(">U0");
            check_ok!("|V0");
            check_ok!("|V7");

            // Presence or absence of units
            check_ok!(">i8");
            check_ok!(">m8[us]");
            check_err!(">i8[us]", MissingOrUnexpectedUnits { .. });
            check_err!(">m8", MissingOrUnexpectedUnits { .. });
        }
    }
}

#[cfg(test)]
#[deny(unused)]
mod tests {
    use super::*;

    #[test]
    fn display_simple() {
        assert_eq!(
            TypeStr {
                endianness: Endianness::Little,
                type_kind: TypeKind::Int,
                size: 8,
                time_units: None,
            }.to_string(),
            "<i8",
        );

        assert_eq!(
            TypeStr {
                endianness: Endianness::Irrelevant,
                type_kind: TypeKind::ByteStr,
                size: 13,
                time_units: None,
            }.to_string(),
            "|S13",
        );

        assert_eq!(
            TypeStr {
                endianness: Endianness::Big,
                type_kind: TypeKind::TimeDelta,
                size: 8,
                time_units: Some(TimeUnits::Nanosecond),
            }.to_string(),
            ">m8[ns]",
        );
    }

    #[test]
    fn roundtrip() {
        macro_rules! check_roundtrip {
            ($text:expr) => {
                let text = $text.to_string();
                match text.parse::<TypeStr>() {
                    Err(e) => panic!("Failed to parse {:?}: {}", text, e),
                    Ok(v) => assert_eq!(text, v.to_string()),
                }
            };
        }

        check_roundtrip!(">i8");
        check_roundtrip!(">f16");
        check_roundtrip!("<i8");
        check_roundtrip!("<i1");
        check_roundtrip!(">i1");
        check_roundtrip!("|i1");
        check_roundtrip!("|S7");
        check_roundtrip!("|S0");
        check_roundtrip!("<S0");
        check_roundtrip!(">U3");
        check_roundtrip!("<m8[D]");
        check_roundtrip!(">m8[ms]");
    }
}
