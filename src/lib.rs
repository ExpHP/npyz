#![warn(missing_docs)]

/*!
Serialize and deserialize the NumPy's
[*.npy binary format](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html).

# Overview

[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

One-dimensional arrays of types that implement the [`Serialize`], [`Deserialize`],
and/or [`AutoSerialize`] traits are supported. These are:

 * primitive types: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`. These map to the `numpy`
   types of `int8`, `uint8`, `int16`, etc.
 * `struct`s annotated as e.g. `#[derive(nippy::Serialize)]`. These map to `numpy`'s
     [Structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html). They can contain the
     following field types:
   * primitive types,
   * other structs that implement the traits,
   * arrays of types that implement the traits (including arrays) of length ≤ 16.
 * `struct`s with manual trait implementations. An example of this can be found in the
   [roundtrip test](https://github.com/ExpHP/nippy/tree/master/tests/roundtrip.rs).

To successfully import an array from NPY using the `#[derive(nippy::Serialize)]` mechanism,
you must enable the `"derive"` feature, and the target struct must contain:

* corresponding number of fields in the same order,
* corresponding names of fields,
* compatible field types.

# Examples

More examples can be found in the [examples](https://github.com/potocpav/npy-rs/tree/master/examples)
directory.

Let's create a simple *.npy file in Python:

```python
import numpy as np
a = np.array([1, 3.5, -6, 2.3])
np.save('examples/plain.npy', a)
```

Now, we can load it in Rust:

```rust
use nippy::NpyReader;

fn main() -> std::io::Result<()> {
    let bytes = std::fs::read("examples/plain.npy")?;

    let data: NpyReader<f64, _> = NpyReader::new(&bytes[..])?;
    for number in data {
        let number = number?;
        eprintln!("{}", number);
    }
    Ok(())
}
```

And we can see our data:

```text
1
3.5
-6
2.3
```

## Reading structs from record arrays

Let us move on to a slightly more complex task. We create a structured array in Python:

```python
import numpy as np
a = np.array([(1,2.5,4), (2,3.1,5)], dtype=[('a', 'i4'),('b', 'f4'),('c', 'i8')])
np.save('examples/simple.npy', a)
```

To load this in Rust, we need to create a corresponding struct.
There are three derivable traits we can define for it:

* [`Deserialize`] — Enables easy reading of `.npy` files.
* [`AutoSerialize`] — Enables easy writing of `.npy` files. (in a default format)
* [`Serialize`] — Supertrait of `AutoSerialize` that allows one to specify a custom [`DType`].

**Enable the `"derive"` feature in `Cargo.toml`,**
and make sure the field names and types all match up:
*/

// It is not currently possible in Cargo.toml to specify that an optional dependency should
// also be a dev-dependency.  Therefore, we discretely remove this example when generating
// doctests, so that:
//    - It always appears in documentation (`cargo doc`)
//    - It is only tested when the feature is present (`cargo test --features derive`)
#![cfg_attr(any(not(doctest), feature="derive"), doc = r##"
```
use nippy::NpyReader;

// make sure to add `features = ["derive"]` in Cargo.toml!
#[derive(nippy::Deserialize, Debug)]
struct Struct {
    a: i32,
    b: f32,
    c: i64,
}

fn main() -> std::io::Result<()> {
    let bytes = std::fs::read("examples/structured.npy")?;

    let data: NpyReader<Struct, _> = NpyReader::new(&bytes[..])?;
    for row in data {
        let row = row?;
        eprintln!("{:?}", row);
    }
    Ok(())
}
```
"##)]
/*!
The output is:

```text
Array { a: 1, b: 2.5, c: 4 }
Array { a: 2, b: 3.1, c: 5 }
```

## Optional features

Implementations of the serialization traits for `num_complex` complex numbers can be enabled
by activating the `complex` feature.
*/

// Reexport the macros.
#[cfg(feature = "derive")] pub use nippy_derive::*;

mod header;
mod read;
mod write;
mod type_str;
mod serialize;

pub use header::{DType, Field};
#[allow(deprecated)]
pub use read::{NpyData, NpyReader, Order};
pub use write::{to_file, OutFile, NpyWriter, Builder};
pub use serialize::{Serialize, Deserialize, AutoSerialize};
pub use serialize::{TypeRead, TypeWrite, TypeWriteDyn, DTypeError};
pub use type_str::{TypeStr, ParseTypeStrError};
