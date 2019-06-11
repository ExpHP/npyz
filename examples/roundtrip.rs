#[macro_use]
extern crate npy_derive;
extern crate npy;

use std::io::Read;

#[derive(Serialize, Deserialize, AutoSerialize, Debug, PartialEq, Clone)]
struct Array {
    a: i32,
    b: f32,
}

fn main() {
    let pi = std::f32::consts::PI;
    let mut arrays = vec![];
    for i in 0..360i32 {
        arrays.push(Array { a: i, b: (i as f32 * pi / 180.0).sin() });
    }

    npy::to_file("examples/roundtrip.npy", arrays).unwrap();

    let mut buf = vec![];
    std::fs::File::open("examples/roundtrip.npy").unwrap()
        .read_to_end(&mut buf).unwrap();

    for (i, arr) in npy::NpyData::from_bytes(&buf).unwrap().into_iter().enumerate() {
        assert_eq!(Array { a: i as i32, b: (i as f32 * pi / 180.0).sin() }, arr);
    }
}
