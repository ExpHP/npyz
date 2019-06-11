#![feature(test)]

extern crate npy;
extern crate test;

use npy::{Serialize, Deserialize, AutoSerialize, TypeWrite, TypeRead};
use test::Bencher;
use test::black_box as bb;

const NITER: usize = 100_000;

macro_rules! gen_benches {
    ($T:ty, $new:expr) => {
        #[inline(never)]
        fn test_data() -> Vec<u8> {
            let mut raw = Vec::new();
            let writer = <$T>::writer(&<$T>::default_dtype()).unwrap();
            for i in 0usize..NITER {
                writer.write_one(&mut raw, &$new(i)).unwrap();
            }
            raw
        }

        #[bench]
        fn read(b: &mut Bencher) {
            let raw = test_data();
            b.iter(|| {
                let dtype = <$T>::default_dtype();
                let reader = <$T>::reader(&dtype).unwrap();

                let mut remainder = &raw[..];
                for _ in 0usize..NITER {
                    let (value, new_remainder) = reader.read_one(remainder);
                    bb(value);
                    remainder = new_remainder;
                }
                assert_eq!(remainder.len(), 0);
            });
        }

        #[bench]
        fn read_to_vec(b: &mut Bencher) {
            // FIXME: Write to a Cursor<Vec<u8>> once #16 is merged
            let path = concat!("benches/bench_", stringify!($T), ".npy");

            npy::to_file(path, (0usize..NITER).map($new)).unwrap();
            let bytes = std::fs::read(path).unwrap();

            b.iter(|| {
                bb(npy::NpyData::<$T>::from_bytes(&bytes).unwrap().to_vec())
            });
        }

        #[bench]
        fn write(b: &mut Bencher) {
            b.iter(|| {
                bb(test_data())
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
