/*!

DType to/from rust type documentation.

> This module does not export any items.  It is used solely as a documentation page.

This page describes all of the numpy datatypes supported by all of the [`Serialize`],
[`Deserialize`], and [`AutoSerialize`] impls provided by this crate.  These descriptions mainly
focus on describing which numpy type character codes are supported by each rust type.

The documentation of numpy's character codes (and its type strings in general) can be found here:
* [numpy `__array_interface__` documentation](https://numpy.org/doc/stable/reference/arrays.interface.html#python-side)

## Simple primitive types

Integers and floats correspond to simple dtypes:

### Integers

* The rust types `i8`, `i16`, `i32`, `i64` use type code `i`.
* The rust types `u8`, `u16`, `u32`, `u64` use type code `u`.

**Notice:** numpy does not support 128-bit integers</li>

### Floats

* The rust types `f32`, `f64` use type code `f`.
* When the **`"half"`** feature is enabled, [`f16`] is also supported.

**Notice:** numpy *does* have 128-bit floats, but it is not currently supported by `npyz`.

### Complex

When the **`"complex"`** feature is enabled, rust types [`Complex32`] and [`Complex64`] may use type code `c`.

**Notice:** numpy does have have complex numbers backed by 128-bit floats, but this is not supported by `npyz`.

### Bool

The rust type `bool` may be serialized as `|b1`.

### Endianness

In all of the above cases, npyz uses the machine endianness by default when serializing, but
supports serializing and deserializing as any endianness.  The size of the datatype in the file
must match the size of the rust type used.

## Date and time

There are two type codes used by numpy for time and date.

* `m`: A `numpy.timedelta64`.
* `M`: A `numpy.datetime64`.

Both of these are represented as 8-byte signed integers, and therefore can use **`i64`** in rust.

The type strings for these types must specify units in square brackets, e.g. `"<m8[ns]"` for
nanoseconds.

## String and blob types

### Overview

There are three type codes for variable-sized strings of data found in npy files:

* `|VN`: A fixed-size array of `N` bytes.
* `|SN` (or `|aN`): A possibly-null-terminated sequence of bytes of length `<= N`.
* `<UN`: An array of `N` Unicode code points, each encoded as a 4-byte integer.

Each will be described in its own section further below.
The following support matrix shows how various rust types may serialize as these type codes.

| rust type               | feature      | `VM` | `SM`/`aM` | `UM` |  [`AutoSerialize`] dtype | notes |
|:---------               | ------------ | --- | ------- | --- | ------------------ | :--- |
| `String`/`str`          |              | ❌ | ✅ | ✅ | ➖   | |
| `Vec<u8>`/`[u8]`        |              | ✅ | ✅ | ❌ | ➖   | length must `== M` when writing `V` |
| `Vec<u32>`/`[u32]`      |              | ❌ | ❌ | ✅ | ➖   | most general type to read `U` |
| `Vec<char>`/`[char]`    |              | ❌ | ❌ | ✅ | ➖   | |
| [`FixedSizeBytes`]`<N>` |              | ✅ | ❌ | ❌ | `VN` | requires `N == M` |
| [`ArrayVec`]`<u8, N>`   | `"arrayvec"` | ✅ | ✅ | ❌ | ➖   | `VM` requires `M <= N` upfront <br/> `S`/`a` truncates when reading |
| [`ArrayVec`]`<u32, N>`  | `"arrayvec"` | ❌ | ❌ | ✅ | `UN` | truncates when reading |
| [`ArrayVec`]`<char, N>` | `"arrayvec"` | ❌ | ❌ | ✅ | `UN` | truncates when reading |
| [`ArrayString`]`<N>`    | `"arrayvec"` | ❌ | ✅ | W  | `SN` | truncates when reading |

Legend:
* ❌: Cannot read or write as this dtype.
* ✅: Can read and write as this dtype.
* ➖: This type does not implement [`AutoSerialize`].
* W: Can be written but not read.

### Raw byte blobs (`|VN`)

This is the simplest sequence type.  It is a blob of exactly `N` bytes.

This type is most easily read as `Vec<u8>`.  However, if `N` is known at compile-time, then
the individual heap allocations per item can be avoided by using [`FixedSizeBytes`] instead;
this is a newtype wrapper around `[u8; N]`.

(you cannot use `[u8; N]` directly because this would be ambiguous in a structured array;
 see the section on "Array members")

### Unicode strings (`<UN`, `>UN`)

This is the type natively used by numpy for Python 3's `str`.

It is an array of `N` Unicode code points, each encoded as a 4-byte integer.
Trailing null code points are not considered to be part of the content. (but interior nulls are)

Notice these are "code points" and not "scalar values"!
One could think of this format as "UTF-32, but surrogates are allowed".
The following Rust types are supported:

* `Vec<u32>`, which is able to read any valid `U` value from a file.
* `Vec<char>`, which will fail on reading surrogates.
* `String`, which will fail on reading surrogates.

Notice that `String` also alternatively supports `|SN` if you want a more compressed representation
in the file, however this is a non-standard convention (see the section on `|SN` for more details).

### Possibly-null-terminated byte blobs (`|SN`)

This is a legacy string type that roughly corresponds to Python 2's `str`.

This is similar to `|VN`, but trailing 0 bytes are not considered to be part of the content.
(Interior null bytes, on the other hand, are part of the content).  numpy itself places no
restrictions on the contents of the bytes.

The most natural type to decode this with is `Vec<u8>`.

However, npyz also supports `String` for this type.  When reading `S` into `String`, the contents
must be valid UTF-8, or reading will fail.  Using `S` for UTF-8 encoded strings is not
a standard practice in python, so beware when using this on files from unknown sources!

### `"arrayvec"` feature

The types described above for `S` and `U` all require individual heap allocations for each item,
which may be costly in terms of memory.  To address this, one may enable the `"arrayvec"` feature
to allow the serialization/deserialization of types from [`arrayvec`].

These types have the additional benefit that they implement [`AutoSerialize`] (by defaulting to the
const parameter `N` as their size), though they all support dtypes with arbitrary size.

When reading a string that is too long to fit in the destination type, it will be truncated.
The justification is that there is no straightforward way for downstream code to implement such
recovery behavior on their own.
Truncation is only performed on `S`, `a`, and `U` data (never on `V` data), and always produces a
valid value of that type.  (e.g. `ArrayString<N>` will truncate to a valid UTF-8 prefix, while
`ArrayVec<u8, N>` will truncate to arbitrary bytes).

If you do not wish to allow truncation to occur, you may check that [`TypeStr::size_field`]`() <= N`.
(`N` being the const length of the arrayvec)

The [`arrayvec`] types support almost all of the same operations as their heap-allocated
counterparts. However, reading `U` into [`ArrayString`] is expressly forbidden, because
due to the transcoding, there is no reliable way for the caller to detect when truncation may
have occurred.

## Structured arrays

One can work with structured arrays by enabling the **`"derive"`** feature, which provides derive
macros for [`Serialize`], [`Deserialize`], and [`AutoSerialize`].
*/
#![cfg_attr(any(not(doctest), feature="derive"), doc = r##"
```
// make sure to add `features = ["derive"]` in Cargo.toml!
#[derive(npyz::Deserialize)]
struct Struct {
    a: i32,
    b: f32,
}
```
"##)]
/*!

This type can be used to deserialize the numpy dtype `np.dtype([('a', '<i4'), ('b', '<f4')])`.

### Array members

Members of structured arrays are allowed to be n-dimensional arrays.  These can be represented
in rust using the primitive array type `[T; N]`:
 */
#![cfg_attr(any(not(doctest), feature="derive"), doc = r##"
```
// make sure to add `features = ["derive"]` in Cargo.toml!
#[derive(npyz::Deserialize)]
struct Struct {
    a: [[i32; 3]; 4],
}
```
"##)]
/*!

This type can deserialize the numpy dtype `np.dtype([('a', '<i4', (4, 3))])`.
**/

#[allow(unused)] // used by docstring
use crate::{FixedSizeBytes, TypeStr, Deserialize, Serialize, AutoSerialize, DType};

#[cfg(feature = "arrayvec")]
#[allow(unused)] // used by docstring
use arrayvec::{self, ArrayVec, ArrayString};

#[cfg(feature = "half")]
#[allow(unused)] // used by docstring
use half::f16;

#[cfg(feature = "complex")]
#[allow(unused)] // used by docstring
use num_complex::{Complex32, Complex64};