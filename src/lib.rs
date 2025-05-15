#![warn(missing_docs)]

/*!
Serialize and deserialize the NumPy's
[*.npy binary format](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html).

# Overview

[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

## Optional cargo features

No features are enabled by default.  Here is the list of existing features:

* There are a couple of features which enable support for serialization/deserialization of foreign
  types. These require opt-in because they can be stability hazards; a major version bump of `npyz`
  may introduce a major version bump of one of these crates.  (NOTE: to ease this issue somewhat,
  `npyz` will re-export the versions of the crates it uses)
  * **`"complex"`** enables the use of [`num_complex::Complex`].
  * **`"half"`** enables the use of [`half::f16`].
  * **`"arrayvec"`** enables the use of [`arrayvec::ArrayVec`] and [`arrayvec::ArrayString`]
    as alternatives to `Vec` and `String` for some string types.
* **`"derive"`** enables derives of traits for working with structured arrays.
* **`"npz"`** enables adapters for working with NPZ files
  (including scipy sparse matrices),
  adding a public dependency on the `zip` crate.
  This requires opt-in because `zip` has a fair number of transitive dependencies.
  (note that some npz-related helper functions are available even without the feature)

## Reading

Let's create a simple *.npy file in Python:

```python
import numpy as np
a = np.array([1, 3.5, -6, 2.3])
np.save('test-data/plain.npy', a)
```

Now, we can load it in Rust using [`NpyFile`]:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("test-data/plain.npy")?;

    // Note: In addition to byte slices, this accepts any io::Read
    let npy = npyz::NpyFile::new(&bytes[..])?;
    for number in npy.data::<f64>()? {
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

### Inspecting properties of the array

[`NpyFile`] provides methods that let you inspect the array.

```rust
fn main() -> std::io::Result<()> {
    let bytes = std::fs::read("test-data/c-order.npy")?;

    let data = npyz::NpyFile::new(&bytes[..])?;
    assert_eq!(data.shape(), &[2, 3, 4]);
    assert_eq!(data.order(), npyz::Order::C);
    assert_eq!(data.strides(), &[12, 4, 1]);

    // convenience method for reading to vec
    println!("{:?}", data.into_vec::<f64>());
    Ok(())
}
```

## Writing

The primary interface for writing npy files is the [`WriterBuilder`] trait.

```rust
use npyz::WriterBuilder;

fn main() -> std::io::Result<()> {
    // Any io::Write is supported.  For this example we'll
    // use Vec<u8> to serialize in-memory.
    let mut out_buf = vec![];
    let mut writer = {
        npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[2, 3])
            .writer(&mut out_buf)
            .begin_nd()?
    };

    writer.push(&100)?;
    writer.push(&101)?;
    writer.push(&102)?;
    // you can also write multiple items at once
    writer.extend(vec![200, 201, 202])?;
    writer.finish()?;

    eprintln!("{:02x?}", out_buf);
    Ok(())
}
```

## Supported dtypes

A complete description of the supported numpy dtypes and the corresponding rust types
can be found on the [`crate::type_matchup_docs`] module.

## Working with `ndarray`

Using the [`ndarray`](https://docs.rs/ndarray) crate?  No problem!
At the time, no conversion API is provided by `npyz`, but one can easily be written:

```rust
// Example of parsing to an array with fixed NDIM.
fn to_array_3<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::Array3<T> {
    use ndarray::ShapeBuilder;

    let shape = match shape[..] {
        [i1, i2, i3] => [i1 as usize, i2 as usize, i3 as usize],
        _  => panic!("expected 3D array"),
    };
    let true_shape = shape.set_f(order == npyz::Order::Fortran);

    ndarray::Array3::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

// Example of parsing to an array with dynamic NDIM.
fn to_array_d<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::ArrayD<T> {
    use ndarray::ShapeBuilder;

    let shape = shape.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    let true_shape = shape.set_f(order == npyz::Order::Fortran);

    ndarray::ArrayD::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

pub fn main() -> std::io::Result<()> {
    let bytes = std::fs::read("test-data/c-order.npy")?;
    let reader = npyz::NpyFile::new(&bytes[..])?;
    let shape = reader.shape().to_vec();
    let order = reader.order();
    let data = reader.into_vec::<i64>()?;

    println!("{:?}", to_array_3(data.clone(), shape.clone(), order));
    println!("{:?}", to_array_d(data.clone(), shape.clone(), order));
    Ok(())
}
```

Likewise, here is a function that can be used to write an ndarray:

```rust
use std::io;
use std::fs::File;

use ndarray::Array;
use npyz::WriterBuilder;

// Example of writing an array with unknown shape.  The output is always C-order.
fn write_array<T, S, D>(writer: impl io::Write, array: &ndarray::ArrayBase<S, D>) -> io::Result<()>
where
    T: Clone + npyz::AutoSerialize,
    S: ndarray::Data<Elem=T>,
    D: ndarray::Dimension,
{
    let shape = array.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();
    let c_order_items = array.iter();

    let mut writer = npyz::WriteOptions::new().default_dtype().shape(&shape).writer(writer).begin_nd()?;
    writer.extend(c_order_items)?;
    writer.finish()
}

pub fn main() -> io::Result<()> {
    let array = Array::from_shape_fn((6, 7, 8), |(i, j, k)| 100*i as i32 + 10*j as i32 + k as i32);
    // even weirdly-ordered axes and non-contiguous arrays are fine
    let view = array.view(); // shape (6, 7, 8), C-order
    let view = view.reversed_axes(); // shape (8, 7, 6), fortran order
    let view = view.slice(ndarray::s![.., .., ..;2]); // shape (8, 7, 3), non-contiguous
    assert_eq!(view.shape(), &[8, 7, 3]);

    let mut file = io::BufWriter::new(File::create("examples/output/ndarray.npy")?);
    write_array(&mut file, &view)
}
```

## Structured arrays

`npyz` supports structured arrays!  Consider the following structured array created in Python:

```python
import numpy as np
a = np.array([(1,2.5,4), (2,3.1,5)], dtype=[('a', 'i4'),('b', 'f4'),('c', 'i8')])
np.save('test-data/simple.npy', a)
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
// make sure to add `features = ["derive"]` in Cargo.toml!
#[derive(npyz::Deserialize, Debug)]
struct Struct {
    a: i32,
    b: f32,
    c: i64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("test-data/structured.npy")?;

    let npy = npyz::NpyFile::new(&bytes[..])?;
    for row in npy.data::<Struct>()? {
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

## `.npz` files

* To work with `.npz` files in general, see the [`npz` module][`npz`].
* To work with `scipy.sparse` matrices see the [`sparse` module][`sparse`].

*/

// Reexport the macros.
#[cfg(feature = "derive")] pub use npyz_derive::*;

mod header;
mod read;
mod write;
mod type_str;
mod serialize;
#[cfg(feature = "npz")]
mod npz_feature;

pub mod npz;
#[cfg(feature = "npz")]
pub mod sparse;

pub mod type_matchup_docs;

// Expose public dependencies
#[cfg(feature = "complex")]
pub use num_complex;
#[cfg(feature = "arrayvec")]
pub use arrayvec;
#[cfg(feature = "npz")]
pub use zip;
#[cfg(feature = "half")]
pub use half;

pub use header::{DType, Field};
#[allow(deprecated)]
pub use read::{NpyData, NpyFile, NpyHeader, NpyReader, Order};
#[allow(deprecated)]
pub use write::{to_file, to_file_1d, OutFile, NpyWriter, write_options, WriteOptions, WriterBuilder};
pub use serialize::FixedSizeBytes;
pub use serialize::{Serialize, Deserialize, AutoSerialize};
pub use serialize::{TypeRead, TypeWrite, TypeWriteDyn, TypeReadDyn, DTypeError};
pub use type_str::{TypeStr, ParseTypeStrError};
pub use type_str::{Endianness, TypeChar, TimeUnits};
