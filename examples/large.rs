
extern crate memmap;
extern crate npy;

use std::fs::File;
use memmap::MmapOptions;

#[derive(npy::Serialize, npy::Deserialize, Debug, Default)]
struct Array {
    a: i32,
    b: f32,
    c: i64,
}

// Use a memory map and iterators, so that this example works for files larger than can fit into RAM
fn main() {
    let file = File::open("examples/simple.npy").unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let data = npy::NpyData::from_bytes(&mmap[..]).unwrap();

    let sum = data.into_iter().fold(Array::default(), |accum, arr: Array| {
        eprintln!("read: {:?}", arr);
        Array { a: accum.a + arr.a, b: accum.b + arr.b, c: accum.c + arr.c }
    });
    eprintln!("sum: {:?}", sum);
}
