# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes yet!

## [0.8.2] - 2023-12-03

### Added
- npyz now has support for half-precision floats using the [`half`](https://crates.io/crates/half) crate.  This support can be enabled with the `"half"` feature.  Thanks, @KeKsBoTer!

## [0.8.1] - 2023-06-05

### Added
- npyz now has experimental support for WASM.  Thanks, @bluenote10!  Not all cargo features are supported/tested yet; please see [this issue](https://github.com/ExpHP/npyz/issues/65#issuecomment-1577409055).

## [0.8.0] - 2023-04-04

### Added
- Added `NpyHeader` which is a more lightweight form of `NpyFile` that isn't a read adapter.  You could use this for instance to parse multiple raw data streams using the same header.

### Changed
- Some of the methods on `NpyFile` have been moved to `NpyHeader`, and are now accessed via `Deref`. If you have been using UFCS to call these methods, you may need to update those callsites.

## [0.7.4] - 2023-02-01

### Added
- `npyz` now uses [`py_literal`](https://crates.io/crates/py_literal) to parse the NPY header, allowing it to support a far greater amount of syntax than before.  Thanks @sudo-carson!
- String metacharacters like `'` in structured array member names are now properly escaped when writing files.
- Dimensions of length 0 in an array member of a structured array are now permitted. (e.g. `dtype=[('a', '>i4', (4,0,4))]`).

## [0.7.3] - 2022-12-06

### Changed
* Writers are now flushed when `.finish()` is called.  Thanks @sirno!

## [0.7.2] - 2022-12-06

This is an extremely minor update that just updates the README.

## [0.7.1] - 2022-09-20

### Added
- `NpyFile` now derives `Clone` for clonable `io::Read`s.

## [0.7.0] - 2022-08-27
### Added
- Added serialization and deserialization for the `U` type.
  Supported rust types are `Vec<char>`, `Vec<u32>`, and `String`
  (and, of course, `[char]`, `[u32]`, and `str` for serialization).
- Added support for the `a` type, which is just an alias of `S`.
- Added the `FixedSizeBytes` wrapper type around `[u8; N]`, which enables
  deserialization of type `V` without individual allocations per element.
- Added the `arrayvec` feature.  This enables deserialization of
  types `U` and `S` without individual allocations per element.
  (thanks to @m-dupont for the initial implementation)
- `NpyReader::shape` and `NpyReader::dtype` accessors, for downstream code without
  access to an `NpyFile`.
- Exposed various data from `TypeStr`.  New types include `Endianness`, `TypeChar`,
  and `TimeUnits` types. `TypeStr` now has getters to extract these fields.
  (thanks to @Shatur for requesting this)

### Changed
- `np.datetime64` now uses i64 instead of u64 for serialization, as it is
  defined to use a symmetric interval around the epoch.
- `DType::num_bytes` has been changed to return `Option<usize>`, rather than panicking when
  the described datatype is too large to fit in the target platform's `usize` type.

## [0.6.0] - 2021-07-05
### Added
- This CHANGELOG.
- Tools for NPZ files and scipy sparse matrices, and the associated **`"npz"`** feature.
- Nested structs inside array fields of a structured array now work.
- `NpyWriter` now allows `T: ?Sized`.

### Changed
- In order to fix nested structs inside array fields, the layout of `DType` had to be changed
  by removing `shape` from `DType::Plain` and adding a new `DType::Array` variant.
- `NpyReader` was split into two types:  You now start by creating an `NpyFile` (which is NOT
  parameterized over `T`), which can then be turned into an `NpyReader`.  The purpose of the
  split is to permit code to make multiple attempts to deserialize into different rust types.
  (e.g. indices in sparse NPZ files may be either `i32` or `i64`)
- `Builder` was changed to a typestate API in order to avoid some runtime panics and to better accomodate NPZ.
  Generally speaking:
  - Code that writes NPY/NPZ files now typically has to `use npyz::WriterBuilder;`.
  - Calls to `npyz::Builder::new()` should now be `npyz::WriteOptions::new()`.
  - `begin_1d()`, `begin_nd()` no longer take arguments.  There are now `.shape(shape)` and
    and `.writer(writer)` methods that must be called to provide these things.

## [0.5.0] - 2021-07-01
### Added
- Initial release of the fork.  The version started at 0.5.0 to indicate that this
  version is a breaking change from 0.4.0 of the original `npy`.
- Adds multidimensional arrays.  *Hallelujah.*
- Adds byteslice and complex types.
- Adds big endian support.
- Adds `NpyReader` for reading from an `io::Read`
- Adds `Builder` and `NpyWriter` for writing to an `io::Write`

[Unreleased]: https://github.com/ExpHP/npyz/compare/0.8.2...HEAD
[0.8.2]: https://github.com/ExpHP/npyz/compare/0.8.1...0.8.2
[0.8.1]: https://github.com/ExpHP/npyz/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/ExpHP/npyz/compare/0.7.4...0.8.0
[0.7.4]: https://github.com/ExpHP/npyz/compare/0.7.3...0.7.4
[0.7.3]: https://github.com/ExpHP/npyz/compare/0.7.2...0.7.3
[0.7.2]: https://github.com/ExpHP/npyz/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/ExpHP/npyz/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/ExpHP/npyz/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/ExpHP/npyz/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/ExpHP/npyz/compare/upstream-0.4.0...0.5.0
