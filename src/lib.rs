#![warn(missing_docs)]

/*!
Serialize and deserialize the NumPy's
[*.npy binary format](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html).

# Overview

[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

One-dimensional arrays of types that implement the [`Serializable`](trait.Serializable.html) trait
are supported. These are:

 * primitive types: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`. These map to the `numpy`
   types of `int8`, `uint8`, `int16`, etc.
 * `struct`s annotated as `#[derive(Serializable)]`. These map to `numpy`'s
     [Structured arrays](https://docs.scipy.org/doc/numpy/user/basics.rec.html). They can contain the
     following field types:
   * primitive types,
   * other [`Serializable`](trait.Serializable.html) structs,
   * arrays of [`Serializable`](trait.Serializable.html) types (including arrays) of length ≤ 16.
 * `struct`s with manual [`Serializable`](trait.Serializable.html) implementations. An example
   this can be found in the [roundtrip test](https://github.com/potocpav/npy-rs/tree/master/tests/roundtrip.rs).

To successfully import an array from NPY using the `#[derive(Serializable)]` mechanism, the target
struct must contain:

* corresponding number of fields in the same order,
* corresponding names of fields,
* compatible field types.
* only little endian fields

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

To load this in Rust, we need to create a corresponding struct, that derives `Serializable`. Make sure
the field names and types all match up:

```
#[macro_use]
extern crate npy_derive;
extern crate npy;

use std::io::Read;
use npy::NpyData;

#[derive(Serializable, Debug)]
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

The output is:

```text
Array { a: 1, b: 2.5, c: 4 }
Array { a: 2, b: 3.1, c: 5 }
```

*/

extern crate byteorder;
#[macro_use]
extern crate nom;

mod header;
mod serializable;
mod npy_data;
mod out_file;

pub use serializable::Serializable;
pub use header::{DType, Field};
pub use npy_data::NpyData;
pub use out_file::{to_file, OutFile, NpyWriter};
