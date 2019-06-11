extern crate npy_derive;
extern crate npy as lol;

#[no_implicit_prelude]
mod not_root {
    use ::npy_derive;

    #[derive(npy_derive::Serialize, npy_derive::Deserialize)]
    struct Struct {
        foo: i32,
        bar: LocalType,
    }

    #[derive(npy_derive::Serialize, npy_derive::Deserialize)]
    struct LocalType;
}

fn main() {}
