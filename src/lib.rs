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

* **`"complex"`** enables parsing of [`num_complex::Complex`]
* **`"derive"`** enables derives of traits for working with structured arrays.

## Reading

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

    // Note: In addition to byte slices, this accepts any io::Read
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

### Inspecting properties of the array

`NpyReader` provides methods that let you inspect the array.

```rust
use nippy::NpyReader;

fn main() -> std::io::Result<()> {
    let bytes = std::fs::read("tests/c-order.npy")?;

    let data: NpyReader<i64, _> = NpyReader::new(&bytes[..])?;
    assert_eq!(data.shape(), &[2, 3, 4]);
    assert_eq!(data.order(), nippy::Order::C);
    assert_eq!(data.strides(), &[12, 4, 1]);
    Ok(())
}
```

## Writing

The primary interface for writing npy files is [`Builder`].

```rust
fn main() -> std::io::Result<()> {
    // Any io::Write is supported.  For this example we'll
    // use Vec<u8> to serialize in-memory.
    let mut out_buf = vec![];
    let mut writer = {
        nippy::Builder::new()
            .default_dtype()
            .begin_nd(&mut out_buf, &[2, 3])?
    };

    writer.push(&100)?; writer.push(&101)?; writer.push(&102)?;
    writer.push(&200)?; writer.push(&201)?; writer.push(&202)?;
    writer.finish()?;

    eprintln!("{:02x?}", out_buf);
    Ok(())
}
```


## Working with `ndarray`

Using the [`ndarray`](https://docs.rs/ndarray) crate?  No problem!
At the time, no conversion API is provided by `nippy`, but one can easily be written:

```rust
use nippy::NpyReader;

// Example of parsing to an array with fixed NDIM.
fn to_array_3<T>(data: Vec<T>, shape: Vec<u64>, order: nippy::Order) -> ndarray::Array3<T> {
    use ndarray::ShapeBuilder;

    let shape = match shape[..] {
        [i1, i2, i3] => [i1 as usize, i2 as usize, i3 as usize],
        _  => panic!("expected 3D array"),
    };
    let true_shape = shape.set_f(order == nippy::Order::Fortran);

    ndarray::Array3::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

// Example of parsing to an array with dynamic NDIM.
fn to_array_d<T>(data: Vec<T>, shape: Vec<u64>, order: nippy::Order) -> ndarray::ArrayD<T> {
    use ndarray::ShapeBuilder;

    let shape = shape.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    let true_shape = shape.set_f(order == nippy::Order::Fortran);

    ndarray::ArrayD::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

fn main() -> std::io::Result<()> {
    let bytes = std::fs::read("tests/c-order.npy")?;
    let reader: NpyReader<i64, _> = NpyReader::new(&bytes[..])?;
    let shape = reader.shape().to_vec();
    let order = reader.order();
    let data = reader.into_vec()?;

    println!("{:?}", to_array_3(data.clone(), shape.clone(), order));
    println!("{:?}", to_array_d(data.clone(), shape.clone(), order));
    Ok(())
}
```

Likewise, here is a function that can be used to write an ndarray:

```rust
use ndarray::Array;
use std::io;
use std::fs::File;

// Example of writing an array with unknown shape.  The output is always C-order.
fn write_array<T, S, D>(writer: impl io::Write, array: &ndarray::ArrayBase<S, D>) -> io::Result<()>
where
    T: Clone + nippy::AutoSerialize,
    S: ndarray::Data<Elem=T>,
    D: ndarray::Dimension,
{
    let shape = array.shape().iter().map(|&x| x as u64).collect::<Vec<_>>();
    let c_order_items = array.iter();

    let mut writer = nippy::Builder::new().default_dtype().begin_nd(writer, &shape)?;
    for item in c_order_items {
        writer.push(item)?;
    }
    writer.finish()
}

fn main() -> io::Result<()> {
    let array = Array::from_shape_fn((6, 7, 8), |(i, j, k)| 100*i as i32 + 10*j as i32 + k as i32);
    // even weirdly-ordered axes and non-contiguous arrays are fine
    let view = array.view(); // shape (6, 7, 8), C-order
    let view = view.reversed_axes(); // shape (8, 7, 6), fortran order
    let view = view.slice(ndarray::s![.., .., ..;2]); // shape (8, 7, 3), non-contiguous
    assert_eq!(view.shape(), &[8, 7, 3]);

    let mut file = io::BufWriter::new(File::create("examples/ndarray-out.npy")?);
    write_array(&mut file, &view)
}
```

## Structured arrays

`nippy` supports structured arrays!  Consider the following structured array created in Python:

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
#[allow(deprecated)]
pub use write::{to_file, to_file_1d, OutFile, NpyWriter, Builder};
pub use serialize::{Serialize, Deserialize, AutoSerialize};
pub use serialize::{TypeRead, TypeWrite, TypeWriteDyn, TypeReadDyn, DTypeError};
pub use type_str::{TypeStr, ParseTypeStrError};
