#[derive(npyz::Serialize, npyz::Deserialize, npyz::AutoSerialize, Debug, PartialEq, Clone)]
struct Struct {
    a: i32,
    b: f32,
}

fn main() -> std::io::Result<()> {
    let pi = std::f32::consts::PI;
    let mut structs = vec![];
    for i in 0..360i32 {
        structs.push(Struct { a: i, b: (i as f32 * pi / 180.0).sin() });
    }

    npyz::to_file_1d("examples/output/roundtrip.npy", structs)?;

    let bytes = std::fs::read("examples/output/roundtrip.npy")?;

    for (i, arr) in npyz::NpyReader::new(&bytes[..]).unwrap().into_iter().enumerate() {
        assert_eq!(Struct { a: i as i32, b: (i as f32 * pi / 180.0).sin() }, arr?);
    }
    Ok(())
}
