// This test is responsible for making sure that the derive macros access

#[no_implicit_prelude]
mod not_root {
    #[derive(::nippy_derive::Serialize, ::nippy_derive::Deserialize, ::nippy_derive::AutoSerialize)]
    struct Struct {
        foo: i32,
        bar: LocalType,
    }

    #[derive(::nippy_derive::Serialize, ::nippy_derive::Deserialize, ::nippy_derive::AutoSerialize)]
    struct LocalType;
}

fn main() {}
