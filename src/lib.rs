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
 * `struct`s annotated as e.g. `#[derive(npy::Serialize)]`. These map to `numpy`'s
     [Structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html). They can contain the
     following field types:
   * primitive types,
   * other structs that implement the traits,
   * arrays of types that implement the traits (including arrays) of length ≤ 16.
 * `struct`s with manual trait implementations. An example of this can be found in the
   [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).

To successfully import an array from NPY using the `#[derive(npy::Serialize)]` mechanism,
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

```
extern crate npy;

use std::io::Read;
use npy::NpyData;

fn main() {
    let mut buf = vec![];
    std::fs::File::open("examples/plain.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<f64> = NpyData::from_bytes(&buf).unwrap();
    for number in data {
        eprintln!("{}", number);
    }
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
#![cfg_attr(any(not(test), feature="derive"), doc = r##"
```
// make sure to add `features = ["derive"]` in Cargo.toml!
extern crate npy;

use std::io::Read;
use npy::NpyData;

#[derive(npy::Deserialize, Debug)]
struct Array {
    a: i32,
    b: f32,
    c: i64,
}

fn main() {
    let mut buf = vec![];
    std::fs::File::open("examples/simple.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<Array> = NpyData::from_bytes(&buf).unwrap();
    for arr in data {
        eprintln!("{:?}", arr);
    }
}
```
"##)]
/*!
The output is:

```text
Array { a: 1, b: 2.5, c: 4 }
Array { a: 2, b: 3.1, c: 5 }
```

*/

// Reexport the macros.
#[cfg(feature = "derive")] extern crate npy_derive;
#[cfg(feature = "derive")] pub use npy_derive::*;

extern crate byteorder;
#[macro_use]
extern crate nom;

mod header;
mod npy_data;
mod out_file;
mod type_str;
mod serialize;

pub use header::{DType, Field};
pub use npy_data::NpyData;
pub use out_file::{to_file, OutFile, NpyWriter};
pub use serialize::{Serialize, Deserialize, AutoSerialize};
pub use serialize::{TypeRead, TypeWrite, TypeWriteDyn, DTypeError};
pub use type_str::{TypeStr, ParseTypeStrError};
