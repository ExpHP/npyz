// This test is responsible for making sure that the derive macros access

#[no_implicit_prelude]
mod not_root {
    #[derive(::npyz_derive::Serialize, ::npyz_derive::Deserialize, ::npyz_derive::AutoSerialize)]
    struct Struct {
        foo: i32,
        bar: LocalType,
    }

    #[derive(::npyz_derive::Serialize, ::npyz_derive::Deserialize, ::npyz_derive::AutoSerialize)]
    struct LocalType;
}

fn main() {}
