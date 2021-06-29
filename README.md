# nippy

Fork of [the `npy` crate](https://github.com/potocpav/npy-rs/issues), under a new name so that I can track my own issues without flooding the original author's repository.  This is where I prototype ideas.

Differences from the latest `npy`:

* Able to read a lot more files by supporting more dtypes.  (e.g. big endian can be read, also strings. I forget what else)
* n-dimensional arrays.  (C-order or fortran order)
  * Writing is done through some [Builder api](https://github.com/ExpHP/nippy/blob/0f3aedcf1a71988af71a8bf8ef7fc65f79924178/src/out_file.rs#L24-L72)
* Outputting to any `io::Write + io::Seek`.
* `io::Seek` is not required when shape is provided.

**Note (2021/05/10):**  It has been two years since I made this fork and tried to submit some PRs to `npy` which are still hanging in limbo.  I'm beginning to wonder if I should pick this back up, clean it up, add proper documentation and publish it to crates.io.

Currently, this is not really in a state to be used in production, but if you are desparate, you can do: 

```toml
[dependencies.npy]
git = "https://github.com/ExpHP/nippy"
rev = "bd608d41f"  # replace with the latest commit
```

Information about the new features may be found by reviewing `cargo doc` and... I guess maybe closed issues on this repo or [this PR comment for the dtype changes](https://github.com/potocpav/npy-rs/pull/15#issuecomment-498839752).  You're somewhat on your own here, until I have time to revisit this proper. Sorry about that!

---

# npy-rs
[![crates.io version](https://img.shields.io/crates/v/npy.svg)](https://crates.io/crates/npy) [![Documentation](https://docs.rs/npy/badge.svg)](https://docs.rs/npy/) [![Build Status](https://travis-ci.org/ExpHP/nippy.svg?branch=master)](https://travis-ci.org/ExpHP/nippy)

Numpy format (*.npy) serialization and deserialization.

<!-- [![Build Status](xxx)](xxx) -->


[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

## Usage

To use **npy-rs**, two dependencies must be specified in `Cargo.toml`:

```toml
npy = "0.4"
npy-derive = "0.4"
```

A typical way to import everything needed is:

```rust
#[macro_use]
extern crate npy_derive;
extern crate npy;
```

The `npy-derive` dependency is only needed for
[structured array](https://docs.scipy.org/doc/numpy/user/basics.rec.html)
serialization.

Data can now be imported from a `*.npy` file:

```rust
use npy::NpyData;

std::fs::File::open("data.npy").unwrap().read_to_end(&mut buf).unwrap();
let data: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();

```

and exported to a `*.npy` file:

```
npy::to_file("data.npy", data).unwrap();
```

See the [documentation](https://docs.rs/npy/) for more information.

Several usage examples are available in the
[examples](https://github.com/potocpav/npy-rs/tree/master/examples) directory; the
[simple](https://github.com/potocpav/npy-rs/blob/master/examples/simple.rs) example shows how to load a file, [roundtrip](https://github.com/potocpav/npy-rs/blob/master/examples/roundtrip.rs) shows both reading
and writing. Large files can be memory-mapped as illustrated in the
[large example](https://github.com/potocpav/npy-rs/blob/master/examples/large.rs).

[Documentation](https://docs.rs/npy/)
