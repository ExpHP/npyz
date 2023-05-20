use std::io::{self,Write,BufWriter,Seek,SeekFrom};
use std::fs::File;
use std::path::Path;
use std::marker::PhantomData;

use byteorder::{WriteBytesExt, LittleEndian};

use crate::serialize::{AutoSerialize, Serialize, TypeWrite};
use crate::header::{self, DType, VersionProps, HeaderSizeType, HeaderEncoding};
use crate::read::Order;

// Long enough to accomodate a large integer followed by ",), }".
// Used when no shape is provided.
const FILLER_FOR_UNKNOWN_SIZE: &'static [u8] = &[b'*'; 19];

struct DataFromBuilder<T: ?Sized> {
    order: Order,
    dtype: DType,
    shape: Option<Vec<u64>>,
    _marker: PhantomData<fn(&T)>, // contravariant
}

pub use write_options::{WriteOptions, WriterBuilder};
pub mod write_options {
    //! Types and traits related to the implementation of [`WriteOptions`].
    //!
    //! There are numerous types and traits involved in the implementation of [`WriteOptions`].
    //! Most of them are not that important; their purpose is to implement a form of typestate.
    //! E.g. you are forbidden from calling [`WriterBuilder::begin_nd`] until you have called
    //! [`WriterBuilder::shape`], [`WriterBuilder::dtype`], and [`WriterBuilder::writer`].
    //!
    //! The most important things from this module ([`WriteOptions`] and [`WriterBuilder`]) are exported
    //! from the root of the crate, so you typically do not need to look in here.
    use super::*;

    /// Represents an almost-empty configuration for an [`NpyWriter`].
    ///
    /// Construction of an [`NpyWriter`] always begins here, with a call to [`WriteOptions::new`].
    /// Then the methods of the [`WriterBuilder`] trait must be used to supply options.
    /// See that trait for more details.
    #[derive(Debug)]
    pub struct WriteOptions<T: ?Sized> {
        order: Order,
        _marker: PhantomData<fn(&T)>, // contravariant
    }

    impl<T: ?Sized> WriteOptions<T> {
        /// Construct an almost empty Writer configuration.
        pub fn new() -> Self { WriteOptions {
            order: Order::C,
            _marker: PhantomData,
        }}
    }

    impl<T: ?Sized> Default for WriteOptions<T> {
        fn default() -> Self { Self::new() }
    }

    impl<T: ?Sized> Clone for WriteOptions<T> {
        fn clone(&self) -> Self { WriteOptions { order: self.order.clone(), _marker: self._marker }}
    }

    /// Trait that provides methods on [`WriteOptions`].
    ///
    /// To obtain an initial instance of this trait, call [`WriteOptions::new`].
    ///
    /// The majority of methods return a type that also implements [`WriterBuilder`]; they are meant
    /// to be chained together to construct a full config.
    pub trait WriterBuilder<T: Serialize + ?Sized>: Sized {
        /// Calls [`Self::dtype`] with the default dtype for the type to be serialized.
        ///
        /// **Calling this method or [`Self::dtype`] is required.**
        fn default_dtype(self) -> WithDType<Self> where T: AutoSerialize { self.dtype(T::default_dtype()) }

        /// Use the specified dtype.
        ///
        /// **Calling `dtype` (or [`Self::default_dtype`]) is required.**
        fn dtype(self, dtype: DType) -> WithDType<Self> { WithDType { inner: self, dtype } }

        /// Set the shape for an n-d array.
        ///
        /// This is required for any array of dimension `!= 1`.
        fn shape(self, shape: &[u64]) -> WithShape<Self> { WithShape { inner: self, shape: shape.to_vec() } }

        /// Set the ouput [`Write`] object.
        ///
        /// **Calling this method is required.**  In some cases (e.g. the builder obtained from an [`NpzWriter`][crate::npz::NpzWriter]),
        /// it will already have been called for you.
        fn writer<W>(self, writer: W) -> WithWriter<W, Self>
        where
            Self: MissingWriter,
        { WithWriter { inner: self, writer } }

        /// Set the data order for arrays with more than one dimension.
        ///
        /// If this is not called, `Order::C` is assumed.
        fn order(self, order: Order) -> Self;

        // getters for properties not encoded in typestate
        #[doc(hidden)] fn __get_order(&self) -> Order;

        /// Begin writing an array of the previously supplied [`shape`][Self::shape].
        fn begin_nd(self) -> io::Result<NpyWriter<T, <Self as HasWriter>::Writer>>
        where
            Self: HasDType + HasWriter + HasShape,
            <Self as HasWriter>::Writer: Write,
        {
            NpyWriter::_begin(DataFromBuilder {
                dtype: self.__get_dtype(),
                order: self.__get_order(),
                shape: Some(self.__get_shape()),
                _marker: PhantomData,
            }, MaybeSeek::Isnt(self.__into_writer()))
        }

        /// Begin writing a 1d array, of length to be inferred from the number of elements written.
        ///
        /// Notice that, in contrast to [`Self::begin_nd`], this method requires [`Seek`].  If you have
        /// a `Vec<u8>`, you can wrap it in an [`io::Cursor`] to satisfy this requirement.
        ///
        /// **Note:** At present, any [`shape`][Self::shape] you *did* happen to provide will be ignored and not
        /// validated against the number of elements written.  This may change in the future.
        fn begin_1d(self) -> io::Result<NpyWriter<T, <Self as HasWriter>::Writer>>
        where
            Self: HasDType + HasWriter,
            <Self as HasWriter>::Writer: Write + Seek,
        {
            NpyWriter::_begin(DataFromBuilder {
                dtype: self.__get_dtype(),
                order: self.__get_order(),
                shape: None,
                _marker: PhantomData,
            }, MaybeSeek::new_seek(self.__into_writer()))
        }
    }

    /// Return type of [`WriterBuilder::writer`].  It represents a config with a known output stream.
    #[derive(Debug, Clone)]
    pub struct WithWriter<W, Builder> {
        pub(super) writer: W,
        pub(super) inner: Builder,
    }

    /// Return type of [`WriterBuilder::dtype`].  It represents a config with a known dtype.
    #[derive(Debug, Clone)]
    pub struct WithDType<Builder> {
        pub(super) dtype: DType,
        pub(super) inner: Builder,
    }

    /// Return type of [`WriterBuilder::shape`].  It represents a config with a known array shape.
    #[derive(Debug, Clone)]
    pub struct WithShape<Builder> {
        pub(super) shape: Vec<u64>,
        pub(super) inner: Builder,
    }

    /// Indicates that a Writer options type includes a DType.
    ///
    /// If you are receiving the following compiler error:
    ///
    /// > ```text
    /// > the trait bound `npyz::WriteOptions<_>: npyz::write_options::HasDType` is not satisfied
    /// > ```
    ///
    /// Then it is most likely because you are missing a call to [`WriterBuilder::dtype`]
    /// or [`WriterBuilder::default_dtype`].
    pub trait HasDType {
        #[doc(hidden)] fn __get_dtype(&self) -> DType;
    }

    /// Indicates that a Writer options type includes a shape.
    ///
    /// If you are receiving the following compiler error:
    ///
    /// > ```text
    /// > the trait bound `npyz::WriteOptions<i64>: npyz::write_options::HasShape` is not satisfied
    /// > ```
    ///
    /// Then it is most likely because you are missing a call to [`WriterBuilder::shape`].
    pub trait HasShape {
        #[doc(hidden)] fn __get_shape(&self) -> Vec<u64>;
    }

    /// Indicates that a Writer options type includes an output stream.
    ///
    /// If you are receiving the following compiler error:
    ///
    /// > ```text
    /// > the trait bound `npyz::WriteOptions<i64>: npyz::write_options::HasWriter` is not satisfied
    /// > ```
    ///
    /// Then it is most likely because you are missing a call to [`WriterBuilder::writer`].
    pub trait HasWriter {
        /// The type of the output stream.
        type Writer;
        #[doc(hidden)]
        fn __into_writer(self) -> Self::Writer;
    }

    // NOTE: This mainly exists to prevent the accidental usage of `.writer()` when working with NPZ files.
    //       (somebody could absent-mindedly copy a piece of Builder code that writes to `Cursor<Vec<u8>>`,
    //        and this wouldn't generate an "unused" warning due to the `&mut cursor`)
    //
    /// Indicates that a Writer options type does not yet have an output stream.
    ///
    /// If you are receiving the following compiler error:
    ///
    /// > ```text
    /// > the trait bound `npyz::write_options::HasWriter<...>: npyz::write_options::MissingWriter` is not satisfied
    /// > ```
    ///
    /// Then it is most likely because you are calling [`WriterBuilder::writer`] on a builder
    /// that already has a writer.  For instance, [`NpzWriter::array`][crate::npz::NpzWriter::array] comes with a writer,
    /// so you do not need to add one.
    pub trait MissingWriter {}

    impl<T: Serialize + ?Sized> WriterBuilder<T> for WriteOptions<T> {
        fn order(mut self, order: Order) -> Self { self.order = order; self }
        fn __get_order(&self) -> Order { self.order }
    }

    impl<W, T: Serialize + ?Sized, B: WriterBuilder<T>> WriterBuilder<T> for WithWriter<W, B> {
        fn order(mut self, order: Order) -> Self { self.inner = self.inner.order(order); self }
        fn __get_order(&self) -> Order { self.inner.__get_order() }
    }

    impl<T: Serialize + ?Sized, B: WriterBuilder<T>> WriterBuilder<T> for WithDType<B> {
        fn order(mut self, order: Order) -> Self { self.inner = self.inner.order(order); self }
        fn __get_order(&self) -> Order { self.inner.__get_order() }
    }

    impl<T: Serialize + ?Sized, B: WriterBuilder<T>> WriterBuilder<T> for WithShape<B> {
        fn order(mut self, order: Order) -> Self { self.inner = self.inner.order(order); self }
        fn __get_order(&self) -> Order { self.inner.__get_order() }
    }

    // Now the silly part where we have to write O(n^2) trait impls
    // Base cases
    impl<B> HasDType for WithDType<B> {
        fn __get_dtype(&self) -> DType { self.dtype.clone() }
    }
    impl<B> HasShape for WithShape<B> {
        fn __get_shape(&self) -> Vec<u64> { self.shape.to_vec() }
    }
    impl<W, B> HasWriter for WithWriter<W, B> {
        type Writer = W;
        fn __into_writer(self) -> Self::Writer { self.writer }
    }
    impl<T: ?Sized> MissingWriter for WriteOptions<T> {}

    // Recursive cases
    macro_rules! forward_typestate_impls {
        ( $(
            ( $inner:tt $impl_generics:tt $Self:tt ): ( $($Trait:ident)* )
        )* ) => {
            $($( forward_typestate_impls!(@single $inner $impl_generics $Self [$Trait]); )*)*
        };

        (@single [$inner:ident] [$($impl_generics:tt)*] [$Self:ty] [HasDType]) => {
            impl<$($impl_generics)*> HasDType for $Self where $inner: HasDType {
                fn __get_dtype(&self) -> DType { self.inner.__get_dtype() }
            }
        };
        (@single [$inner:ident] [$($impl_generics:tt)*] [$Self:ty] [HasShape]) => {
            impl<$($impl_generics)*> HasShape for $Self where $inner: HasShape {
                fn __get_shape(&self) -> Vec<u64> { self.inner.__get_shape() }
            }
        };
        (@single [$inner:ident] [$($impl_generics:tt)*] [$Self:ty] [HasWriter]) => {
            impl<$($impl_generics)*> HasWriter for $Self where $inner: HasWriter {
                type Writer = $inner::Writer;
                fn __into_writer(self) -> Self::Writer { self.inner.__into_writer() }
            }
        };
        (@single [$inner:ident] [$($impl_generics:tt)*] [$Self:ty] [MissingWriter]) => {
            impl<$($impl_generics)*> MissingWriter for $Self where $inner: MissingWriter { }
        };
    }

    forward_typestate_impls!{
        ([B] [B] [WithShape<B>]): (/*HasShape*/ HasDType HasWriter MissingWriter)
        ([B] [B] [WithDType<B>]): (HasShape /*HasDType*/ HasWriter MissingWriter)
        ([B] [W, B] [WithWriter<W, B>]): (HasShape HasDType /*HasWriter MissingWriter*/)
    }
}

/// Interface for writing an NPY file to a data stream.
///
/// To construct an instance of this, you must go through the [`WriterBuilder`] trait:
///
/// <!-- example so nobody has an excuse for not seeing the link to WriterBuilder -->
/// ```rust
/// use npyz::WriterBuilder;
///
/// fn main() -> std::io::Result<()> {
///     // Any io::Write is supported.  For this example we'll
///     // use Vec<u8> to serialize in-memory.
///     let mut out_buf = vec![];
///     let mut writer = {
///         npyz::WriteOptions::new()
///             .default_dtype()
///             .shape(&[2, 3])
///             .writer(&mut out_buf)
///             .begin_nd()?
///     };
///
///     writer.push(&100)?;
///     writer.push(&101)?;
///     writer.push(&102)?;
///     // can write elements from iterators too
///     writer.extend(vec![200, 201, 202])?;
///     writer.finish()?;
///
///     eprintln!("{:02x?}", out_buf);
///     Ok(())
/// }
/// ```
pub struct NpyWriter<Row: Serialize + ?Sized, W: Write> {
    start_pos: Option<u64>,
    shape_info: ShapeInfo,
    num_items: u64,
    fw: MaybeSeek<W>,
    writer: <Row as Serialize>::TypeWriter,
    version_props: VersionProps,
}

enum ShapeInfo {
    // No shape was written; we'll return to write a 1D shape on `finish()`.
    Automatic { offset_in_header_text: u64 },
    // The complete shape has already been written.
    // Raise an error on `finish()` if the wrong number of elements is given.
    Known { expected_num_items: u64 },
}

/// [`NpyWriter`] that writes an entire file.
#[deprecated(since = "0.5.0", note = "Doesn't carry its weight.  Use to_file_1d instead, or replicate the original behavior with Builder::new().default_dtype().begin_1d(std::io::BufWriter::new(std::fs::File::create(path)?))")]
pub type OutFile<Row> = NpyWriter<Row, BufWriter<File>>;

#[allow(deprecated)]
impl<Row: AutoSerialize> OutFile<Row> {
    /// Create a file, using the default format for the given type.
    #[deprecated(since = "0.5.0", note = "Doesn't carry its weight.  Use to_file_1d instead, or replicate the original behavior with Builder::new().default_dtype().begin_1d(std::io::BufWriter::new(std::fs::File::create(path)?))")]
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        WriteOptions::new()
            .default_dtype()
            .writer(BufWriter::new(File::create(path)?))
            .begin_1d()
    }
}

#[allow(deprecated)]
impl<Row: Serialize> OutFile<Row> {
    /// Finish writing the file and close it.  Alias for [`NpyWriter::finish`].
    ///
    /// If omitted, the file will be closed on drop automatically, ignoring any errors
    /// encountered during the process.
    #[deprecated(since = "0.5.0", note = "use .finish() instead")]
    pub fn close(self) -> io::Result<()> {
        self.finish()
    }
}

impl<Row: Serialize + ?Sized , W: Write> NpyWriter<Row, W> {
    fn _begin(builder: DataFromBuilder<Row>, mut fw: MaybeSeek<W>) -> io::Result<Self> {
        let DataFromBuilder { dtype, order, shape, _marker } = builder;

        let start_pos = match fw {
            MaybeSeek::Is(ref mut fw) => Some(fw.seek(SeekFrom::Current(0))?),
            MaybeSeek::Isnt(_) => None,
        };

        if let DType::Array(..) = dtype {
            panic!("the outermost dtype cannot be an array (got: {:?})", dtype);
        }

        let (dict_text, shape_info) = create_dict(&dtype, order, shape.as_deref());
        let (header_text, version, version_props) = determine_required_version_and_pad_header(dict_text);

        fw.write_all(&[0x93u8])?;
        fw.write_all(b"NUMPY")?;
        fw.write_all(&[version.0, version.1])?;

        assert_eq!((header_text.len() + version_props.bytes_before_text()) % 16, 0);
        match version_props.header_size_type {
            HeaderSizeType::U16 => {
                assert!(header_text.len() <= u16::MAX as usize);
                fw.write_u16::<LittleEndian>(header_text.len() as u16)?;
            },
            HeaderSizeType::U32 => {
                assert!(header_text.len() <= u32::MAX as usize);
                fw.write_u32::<LittleEndian>(header_text.len() as u32)?;
            },
        }
        fw.write_all(&header_text)?;

        let writer = match Row::writer(&dtype) {
            Ok(writer) => writer,
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        };

        Ok(NpyWriter {
            start_pos,
            shape_info,
            num_items: 0,
            fw,
            writer,
            version_props,
        })
    }

    /// Append a single row to the file
    pub fn push(&mut self, row: &Row) -> io::Result<()> {
        self.num_items += 1;
        self.writer.write_one(&mut self.fw, row)
    }

    /// Write an iterator to the file
    pub fn extend(&mut self, rows: impl IntoIterator<Item=Row>) -> io::Result<()> where Row: Sized {
        rows.into_iter().try_for_each(|row| self.push(&row))
    }

    fn finish_(&mut self) -> io::Result<()> {
        match self.shape_info {
            ShapeInfo::Known { expected_num_items } => {
                if expected_num_items != self.num_items {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, {
                        format!("shape has {} item(s), but {} item(s) were written!", expected_num_items, self.num_items)
                    }));
                }
            },
            ShapeInfo::Automatic { offset_in_header_text } => {
                // Write the size to the header
                let shape_pos = self.start_pos.unwrap() + self.version_props.bytes_before_text() as u64 + offset_in_header_text;
                let end_pos = self.fw.seek(SeekFrom::Current(0))?;

                self.fw.seek(SeekFrom::Start(shape_pos))?;
                let length = format!("{}", self.num_items);
                self.fw.write_all(length.as_bytes())?;
                self.fw.write_all(&b",), }"[..])?;
                self.fw.write_all(&::std::iter::repeat(b' ').take(FILLER_FOR_UNKNOWN_SIZE.len() - length.len()).collect::<Vec<_>>())?;
                self.fw.seek(SeekFrom::Start(end_pos))?;
            },
        }
        self.fw.flush()?;
        Ok(())
    }

    /// Finish writing the file.
    ///
    /// If no shape was provided, this will update the header to reflect the number of
    /// elements written. If a shape was provided and the number of inserted elements is
    /// incorrect, an error is returned.
    ///
    /// This is automatically called on drop, but in that case, errors are ignored.
    pub fn finish(mut self) -> io::Result<()> {
        self.finish_()
    }
}

fn create_dict(dtype: &DType, order: Order, shape: Option<&[u64]>) -> (Vec<u8>, ShapeInfo) {
    let mut header: Vec<u8> = vec![];
    header.extend(&b"{'descr': "[..]);
    header.extend(dtype.descr().as_bytes());
    header.extend(&b", 'fortran_order': "[..]);
    match order {
        Order::C => header.extend(&b"False"[..]),
        Order::Fortran => header.extend(&b"True"[..]),
    }
    header.extend(&b", 'shape': ("[..]);
    let shape_info = match shape {
        Some(shape) => {
            for x in shape {
                write!(header, "{}, ", x).unwrap();
            }
            header.extend(&b"), }"[..]);
            ShapeInfo::Known { expected_num_items: shape.iter().product() }
        },
        None => {
            let shape_offset = header.len() as u64;
            header.extend(FILLER_FOR_UNKNOWN_SIZE);
            header.extend(&b",), }"[..]);
            ShapeInfo::Automatic { offset_in_header_text: shape_offset }
        },
    };
    (header, shape_info)
}

impl<Row: Serialize + ?Sized, W: Write> Drop for NpyWriter<Row, W> {
    fn drop(&mut self) {
        let _ = self.finish_(); // Ignore the errors
    }
}

/// This does two things:
///
/// - Get the minimum version required to write a file, based on its header text.
/// - Pad the end of the header text so that the data begins aligned to 16 bytes.
///
/// Why must it do these together?  It turns out there's a tricky corner-case interaction for
/// header lengths close to but *just under* 65536, where the padding can push the length over
/// the 65536 threshold, causing version 2 to be used and therefore use 2 additional bytes.
/// Those additional bytes in turn could throw off the padding.
fn determine_required_version_and_pad_header(mut header_utf8: Vec<u8>) -> (Vec<u8>, (u8, u8), VersionProps) {
    use HeaderSizeType::*;
    use HeaderEncoding::*;

    // I'm almost 100% certain that, when regarding the initial length of dict_utf8,
    // there is a precise value at which the optimal version suddenly switches from 1 to 2.
    // I think it is either 65524, 65525, or 65526; just not sure which.  (the newline makes it weird)
    //
    // Unfortunately testing this edge case is not easy, so to be safe we'll give ourselves more wiggle
    // room than could possibly ever be affected by padding and/or pre-header bytes.   - ExpHP
    const SAFE_U16_CUTOFF: usize = 0xffff_fc00; // = 0x1_0000_0000 - 0x400 (which doesn't compile on WASM)

    let required_props = VersionProps {
        header_size_type: if header_utf8.len() >= SAFE_U16_CUTOFF { U32 } else { U16 },
        encoding: if header_utf8.iter().any(|b| !b.is_ascii()) { Utf8 } else { Ascii },
    };

    let version = header::get_minimal_version(required_props);

    // Actual props may differ from required props.  (e.g. if it has unicode, then it needs
    // to use version 3 which will cause the size to be upgraded to U32 even if not needed)
    let actual_props = header::get_version_props(version).expect("generated internally so must be valid");

    // Now pad using the final choice of version.
    //
    // From the numpy documentation:
    //
    //    It is terminated by a newline (\n) and padded with spaces (\x20) to make the total of
    //    len(magic string) + 2 + len(length) + HEADER_LEN be evenly divisible by 64 for alignment purposes.
    const ALIGN_TO: usize = 64;

    let bytes_before_text = actual_props.bytes_before_text();
    header_utf8.extend(&::std::iter::repeat(b' ').take(ALIGN_TO - 1 - ((header_utf8.len() + bytes_before_text) % ALIGN_TO)).collect::<Vec<_>>());
    header_utf8.push(b'\n');
    assert_eq!((header_utf8.len() + bytes_before_text) % ALIGN_TO, 0);

    (header_utf8, version, actual_props)
}

#[deprecated(since = "0.5.0", note = "renamed to to_file_1d")]
/// Serialize an iterator over a struct to a NPY file.
///
/// This only saves a 1D array.  To save an ND array, **you must use the [`WriterBuilder`] API.**
pub fn to_file<S, T, P>(filename: P, data: T) -> std::io::Result<()>
where
    P: AsRef<Path>,
    S: AutoSerialize,
    T: IntoIterator<Item=S>,
{
    to_file_1d(filename, data)
}

/// Serialize an iterator over a struct to a NPY file.
///
/// This only saves a 1D array.  To save an ND array, **you must use the [`WriterBuilder`] API.**
pub fn to_file_1d<S, T, P>(filename: P, data: T) -> std::io::Result<()>
where
    P: AsRef<Path>,
    S: AutoSerialize,
    T: IntoIterator<Item=S>,
{
    #![allow(deprecated)]
    let mut of = OutFile::open(filename)?;
    for row in data {
        of.push(&row)?;
    }
    of.close()
}

// module encapsulating the unsafety of MaybeSeek
use maybe_seek::MaybeSeek;
mod maybe_seek {
    use super::*;

    // A writer that implements Seek even if W doesn't (by potentially panicking).
    //
    // The peculiar design of this is explained in greater detail here:
    // https://github.com/ExpHP/npyz/issues/42#issuecomment-873263846
    pub(crate) enum MaybeSeek<W> {
        // a trait object is needed to smuggle in access to Seek methods without every method needing a Seek bound
        Is(Box<dyn WriteSeek<W>>),
        Isnt(W),
    }

    // Note: W = Self.  It is used to allow W to stand in for the "lifetime" of W
    //       when casting W to a trait object.
    pub(crate) trait WriteSeek<W>: Write + Seek + sealed::Sealed<W> {}

    mod sealed {
        use super::*;

        pub(crate) trait Sealed<W> {}
        impl<W: Write + Seek> Sealed<W> for W {}
    }

    impl<W: Write + Seek> WriteSeek<W> for W {}

    impl<W: Write> Write for MaybeSeek<W> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            match self {
                MaybeSeek::Is(w) => (*w).write(buf),
                MaybeSeek::Isnt(w) => w.write(buf),
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            match self {
                MaybeSeek::Is(w) => (*w).flush(),
                MaybeSeek::Isnt(w) => w.flush(),
            }
        }
    }

    impl<W> Seek for MaybeSeek<W> {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            match self {
                MaybeSeek::Is(w) => (*w).seek(pos),
                MaybeSeek::Isnt(_) => unreachable!("(BUG!) .seek() called on MaybeSeek::Isnt!"),
            }
        }
    }

    impl<W: WriteSeek<W>> MaybeSeek<W> {
        pub fn new_seek(w: W) -> Self {
            let inner = unsafe {
                // The Self type is W, so all lifetime information contained in the unnamed
                // lifetime here is also contained in W.
                //
                // Because `dyn WriteSeek<W> + '_` is invariant in W, the compiler will
                // conservatively assume that it carries all borrows held by W; just as if
                // we *hadn't* erased the lifetime.
                //
                // See discussion here:
                //   https://users.rust-lang.org/t/a-trait-object-with-an-implied-lifetime/29340
                std::mem::transmute::<
                    Box<dyn WriteSeek<W> + '_>,
                    Box<dyn WriteSeek<W> + 'static>,
                >(Box::new(w))
            };
            MaybeSeek::Is(inner)
        }
    }
}

/// Quick API for writing a 1D array to a vector of bytes.
#[cfg(test)]
pub(crate) fn to_bytes_1d<T: AutoSerialize>(data: &[T]) -> io::Result<Vec<u8>> {
    let mut cursor = io::Cursor::new(vec![]);
    to_writer_1d(&mut cursor, data)?;
    Ok(cursor.into_inner())
}

/// Quick API for writing a 1D array to an io::Write.
#[cfg(test)]
pub(crate) fn to_writer_1d<W: io::Write + io::Seek, T: AutoSerialize>(writer: W, data: &[T]) -> io::Result<()> {
    // we might change this later and/or remove the Seek bound from the current function, but for now this will do
    to_writer_1d_with_seeking(writer, data)
}

/// Quick API for writing an n-d array to an io::Write.
#[cfg(test)]
pub(crate) fn to_writer_nd<W: io::Write, T: AutoSerialize>(writer: W, data: &[T], shape: &[u64]) -> io::Result<()> {
    let mut writer = WriteOptions::new().default_dtype().writer(writer).shape(shape).begin_nd()?;
    writer.extend(data)?;
    writer.finish()
}

/// Quick API for writing a 1D array to an io::Write in a manner which makes use of io::Seek.
///
/// (tests will use this instead of 'to_writer_1d' if their purpose is to test the correctness of seek behavior,
/// so that changing 'to_writer_1d' to be Seek-less won't affect these tests)
#[cfg(test)]
pub(crate) fn to_writer_1d_with_seeking<W: io::Write + io::Seek, T: AutoSerialize>(writer: W, data: &[T]) -> io::Result<()> {
    let mut writer = WriteOptions::new().default_dtype().writer(writer).begin_1d()?;
    writer.extend(data)?;
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};
    use crate::NpyFile;

    fn bytestring_contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        haystack.windows(needle.len()).any(move |w| w == needle)
    }

    #[test]
    fn write_1d_simple() -> io::Result<()> {
        let raw_buffer = to_bytes_1d(&[1.0, 3.0, 5.0])?;

        let reader = NpyFile::new(&raw_buffer[..])?;
        assert_eq!(reader.into_vec::<f64>()?, vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn write_1d_in_the_middle() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let prefix = b"lorem ipsum dolor sit amet.";
        let suffix = b"and they lived happily ever after.";

        // write to the cursor both before and after writing the file
        cursor.write_all(prefix)?;
        to_writer_1d_with_seeking(&mut cursor, &[1.0, 3.0, 5.0])?;
        cursor.write_all(suffix)?;

        // check that the seeking did not interfere with our extra writes
        let raw_buffer = cursor.into_inner();
        assert!(raw_buffer.starts_with(prefix));
        assert!(raw_buffer.ends_with(suffix));

        // check the bytes written by `OutFile`
        let written_bytes = &raw_buffer[prefix.len()..raw_buffer.len() - suffix.len()];
        let reader = NpyFile::new(&written_bytes[..])?;
        assert_eq!(reader.into_vec::<f64>()?, vec![1.0, 3.0, 5.0]);

        Ok(())
    }

    #[test]
    fn implicit_finish() -> io::Result<()> {
        let mut cursor = Cursor::new(vec![]);

        let mut writer = WriteOptions::new().default_dtype().writer(&mut cursor).begin_1d()?;
        writer.extend(vec![1.0, 3.0, 5.0, 7.0])?;
        // don't call finish
        drop(writer);

        // check that the shape was written
        let raw_buffer = cursor.into_inner();
        println!("{:?}", raw_buffer);
        assert!(bytestring_contains(&raw_buffer, b"'shape': (4,"));

        Ok(())
    }

    #[test]
    fn write_nd_simple() -> io::Result<()> {
        let mut buffer = vec![];
        to_writer_nd(&mut buffer, &[00, 01, 02, 10, 11, 12], &[2, 3])?;

        let reader = NpyFile::new(&buffer[..])?;
        assert_eq!(reader.shape(), &[2, 3][..]);
        assert_eq!(reader.into_vec::<i32>()?, vec![00, 01, 02, 10, 11, 12]);

        Ok(())
    }

    #[test]
    fn write_nd_wrong_len() -> io::Result<()> {
        let try_writing = |elems: &[i32]| -> io::Result<()> {
            let mut buf = vec![];
            let mut writer = WriteOptions::new().default_dtype().writer(&mut buf).shape(&[2, 3]).begin_nd()?;
            for &x in elems {
                writer.push(&x)?;
            }
            writer.finish()?;
            Ok(())
        };
        assert!(try_writing(&[00, 01, 02, 10, 11]).is_err());
        assert!(try_writing(&[00, 01, 02, 10, 11, 12]).is_ok());
        assert!(try_writing(&[00, 01, 02, 10, 11, 12, 20]).is_err());

        Ok(())
    }
}
