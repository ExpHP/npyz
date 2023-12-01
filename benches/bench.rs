use npyz::AutoSerialize;
use bencher::{Bencher, black_box as bb};
use std::io::Cursor;
use npyz::WriterBuilder;

const NITER: usize = 100_000;

macro_rules! gen_benches {
    // HACK even though we grouped things into modules we have to manually supply names for the
    //      functions because the bencher crate doesn't show the full module path
    ($read_to_vec_testname:ident, $write_testname:ident, $T:ty, $new:expr) => {
        #[inline(never)]
        fn write_array_via_push() -> Vec<u8> {
            let cap = 1000 + <$T>::default_dtype().num_bytes().unwrap() * NITER;
            let mut cursor = Cursor::new(Vec::with_capacity(cap));
            {
                let mut writer = npyz::WriteOptions::new().default_dtype().writer(&mut cursor).begin_1d().unwrap();
                for i in 0usize..NITER {
                    writer.push(&$new(i)).unwrap();
                }
                writer.finish().unwrap();
            }
            cursor.into_inner()
        }

        #[allow(deprecated)]
        fn $read_to_vec_testname(b: &mut Bencher) {
            let bytes = write_array_via_push();
            b.iter(|| {
                bb(npyz::NpyData::<$T>::from_bytes(&bytes).unwrap().to_vec())
            });
        }

        fn $write_testname(b: &mut Bencher) {
            b.iter(|| {
                bb(write_array_via_push())
            });
        }

        bencher::benchmark_group!(benches, $write_testname, $read_to_vec_testname);
    };
}

#[cfg(feature = "derive")]
mod simple {
    use super::*;

    #[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct Simple {
        a: i32,
        b: f32,
    }

    gen_benches!(
        simple_read_to_vec, simple_write,
        Simple, |i| Simple { a: i as i32, b: i as f32 }
    );
}

#[cfg(feature = "derive")]
mod one_field {
    use super::*;

    #[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct OneField {
        a: i32,
    }

    gen_benches!(
        one_field_read_to_vec, one_field_write,
        OneField, |i| OneField { a: i as i32 }
    );
}

#[cfg(feature = "derive")]
mod array {
    use super::*;

    #[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize)]
    #[derive(Debug, PartialEq)]
    struct Array {
        a: [f32; 8],
    }

    gen_benches!(
        array_read_to_vec, array_write,
        Array, |i| Array { a: [i as f32; 8] }
    );
}

mod plain_f32 {
    use super::*;

    gen_benches!(
        plain_f32_read_to_vec, plain_f32_write,
        f32, |i| i as f32
    );
}

#[cfg(feature = "derive")]
bencher::benchmark_main!(plain_f32::benches, array::benches, simple::benches, one_field::benches);
#[cfg(not(feature = "derive"))]
bencher::benchmark_main!(plain_f32::benches);
