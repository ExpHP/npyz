# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added serialization and deserialization for the `U` type.
  Supported rust types are `Vec<char>`, `Vec<u32>`, and `String`
  (and, of course, `[char]`, `[u32]`, and `str` for serialization).
- Added support for the `a` type, which is just an alias of `S`.
- Added the `arrayvec` feature.  This enables deserialization of
  type `U` without individual allocations per element.
- Exposed various data from `TypeStr`.  New types include `Endianness`, `TypeChar`,
  and `TimeUnits` types. `TypeStr` now has getters to extract these fields.

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
- [`NpyWriter`] now allows `T: ?Sized`.

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

[Unreleased]: https://github.com/ExpHP/npyz/compare/0.6.0...HEAD
[0.6.0]: https://github.com/ExpHP/npyz/compare/0.5.0...0.6.0
[0.5.0]: https://github.com/ExpHP/npyz/compare/upstream-0.4.0...0.5.0
