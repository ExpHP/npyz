# nippy

Fork of [the `npy` crate](https://github.com/potocpav/npy-rs/issues), to be published to crates.io under a new name because I got tired after years of radio silence from the original maintainer.

Differences from `npy 0.4`:

* More dtypes
  * Big endian; you can parse `>i4` to `i32`
  * Null-terminated bytestrings `|Sn` and raw bytes `|Vn` to `Vec<u8>`
  * `c16` and `c32` to [Complex](https://docs.rs/num-complex/0.4.0/num_complex/struct.Complex.html) (when enabling the `"complex"` feature)
* n-dimensional arrays
* Proper standard library integration
  * Writing to `io::Write`
  * Reading from `io::Read`
  * `io::Seek` not required! (in most cases)

>  **Note: 2021/07/01.** The time is now.  Once again I've needed this for my own projects, and now I've been working hard to prepare it for release.  Hopefully, I will have succeeded and removed this message long before you ever have the chance to read it.
>
> So.  If you *are* reading this message, and 2021/07/01 was a while ago, then.... well, that sucks.  If you desparately need one of the above features, you can use this as a git dependency for now, and please drop me a message reminding me to come back and finish what I started.
>
> ```toml
> [dependencies.nippy]
> git = "https://github.com/ExpHP/nippy"
> rev = "bd608d41f"  # replace with the latest commit
> ```

---

# npy-rs
[![crates.io version](https://img.shields.io/crates/v/nippy.svg)](https://crates.io/crates/nippy) [![Documentation](https://docs.rs/nippy/badge.svg)](https://docs.rs/nippy/) [![Build Status](https://travis-ci.org/ExpHP/nippy.svg?branch=master)](https://travis-ci.org/ExpHP/nippy)

Numpy format (*.npy) serialization and deserialization.

<!-- [![Build Status](xxx)](xxx) -->


[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

## Usage

To use **nippy**, two dependencies must be specified in `Cargo.toml`:

```toml
npy = {version = "0.5", features = ["derive"]}
```

A typical way to import everything needed is:

The `npy-derive` dependency is only needed for
[structured array](https://docs.scipy.org/doc/numpy/user/basics.rec.html)
serialization.

Data can now be imported from a `*.npy` file:

```rust
use nippy::NpyData;

std::fs::File::open("data.npy").unwrap().read_to_end(&mut buf).unwrap();
let data: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();

```

and exported to a `*.npy` file:

```
nippy::to_file("data.npy", data).unwrap();
```

See the [documentation](https://docs.rs/nippy/) for more information.

Several usage examples are available in the
[examples](https://github.com/ExpHP/nippy/tree/master/examples) directory; the
[structured](https://github.com/ExpHP/nippy/blob/master/examples/structured.rs) example shows how to load a structured array, [roundtrip](https://github.com/ExpHP/nippy/blob/master/examples/roundtrip.rs) shows both reading
and writing.

[Documentation](https://docs.rs/nippy/)
