#![feature(test)]

extern crate test;

use npy::{AutoSerialize, NpyWriter};
use test::Bencher;
use test::black_box as bb;
use std::io::Cursor;

const NITER: usize = 100_000;

macro_rules! gen_benches {
    ($T:ty, $new:expr) => {
        #[inline(never)]
        fn write_array_via_push() -> Vec<u8> {
            let cap = 1000 + <$T>::default_dtype().num_bytes() * NITER;
            let mut cursor = Cursor::new(Vec::with_capacity(cap));
            {
                let mut writer = NpyWriter::begin(&mut cursor).unwrap();
                for i in 0usize..NITER {
                    writer.push(&$new(i)).unwrap();
                }
                writer.finish().unwrap();
            }
            cursor.into_inner()
        }

        #[allow(deprecated)]
        #[bench]
        fn read_to_vec(b: &mut Bencher) {
            let bytes = write_array_via_push();
            b.iter(|| {
                bb(npy::NpyData::<$T>::from_bytes(&bytes).unwrap().to_vec())
            });
        }

        #[bench]
        fn write(b: &mut Bencher) {
            b.iter(|| {
                bb(write_array_via_push())
            });
        }
    };
}

#[cfg(feature = "derive")]
mod simple {
    use super::*;

    #[derive(npy::Serialize, npy::Deserialize, npy::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct Simple {
        a: i32,
        b: f32,
    }

    gen_benches!(Simple, |i| Simple { a: i as i32, b: i as f32 });
}

#[cfg(feature = "derive")]
mod one_field {
    use super::*;

    #[derive(npy::Serialize, npy::Deserialize, npy::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct OneField {
        a: i32,
    }

    gen_benches!(OneField, |i| OneField { a: i as i32 });
}

#[cfg(feature = "derive")]
mod array {
    use super::*;

    #[derive(npy::Serialize, npy::Deserialize, npy::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct Array {
        a: [f32; 8],
    }

    gen_benches!(Array, |i| Array { a: [i as f32; 8] });
}

mod plain_f32 {
    use super::*;

    gen_benches!(f32, |i| i as f32);
}
