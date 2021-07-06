//! Utilities for working with `npz` files.
//!
//! NPZ files are a format used by numpy to store archives of multiple arrays.
//! The format itself is just a `.zip` file containing files with a `.npy` extension.
//!
//! When the `"npz"` feature is enabled, this module provides adapters around the
//! [`zip`][::zip] crate that read and write NPZ files.
//!
//! Even without the `"npz"` feature, this module always provides a set of utility functions
//! for converting between array names and filenames inside the zip file.
//! This allows you to use your own zip library. The [`examples` directory] includes an example
//! of how to read and write an npz without the feature.
//!
//! [`examples` directory]: https://github.com/ExpHP/npyz/tree/master/examples

#[cfg(feature = "npz")]
pub use crate::npz_feature::*;

/// Get the name of the array that would correspond to the given filename inside a zip file.
///
/// This tries to match the behavior of numpy's own npz-loading behavior:
///
/// * Case sensitive (only `.npy` files, not `.NPY` files)
/// * Allows weird characters like `/` and `.`.  Makes no attempt to normalize paths.
/// * Treats an interior null as the end of the path.  Notice that this means that multiple
///   different filenames could produce the same array name in a maliciously constructed zip.
///
/// Returns `None` if numpy would not consider the file to be an array.
///
/// _This function does not require any cargo features._
pub fn array_name_from_file_name(path_in_zip: &str) -> Option<&str> {
    let mut path = path_in_zip;
    if let Some(idx) = path.find("\0") {
        path = &path[..idx];
    }

    if path.ends_with(".npy") {
        Some(&path[..path.len() - 4])
    } else {
        None
    }
}

/// Get the filename in a zip that `np.savez` would use for a keyword argument.
///
/// **Note:** This does accept the name `"file"`, even though this cannot normally be used in `np.savez`
/// due to technical limitations. (numpy can read the file just fine)
///
/// _This function does not require any cargo features._
pub fn file_name_from_array_name(name: &str) -> String {
    format!("{}.npy", name)
}

/// Get the filename in a zip that would be used for an array supplied as a positional
/// argument to `np.savez`.
///
/// _This function does not require any cargo features._
pub fn file_name_from_index(index: i32) -> String {
    format!("arr_{}.npy", index)
}

