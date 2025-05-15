#![recursion_limit = "256"]

/*!
Derive `npyz`'s traits for structured arrays.

Using this crate, it is enough to `#[derive(npyz::Serialize, npyz::Deserialize)]` on a struct to be able to
serialize and deserialize it. All of the fields must implement [`Serialize`](../npyz/trait.Serialize.html)
and [`Deserialize`](../npyz/trait.Deserialize.html) respectively.

*/

use proc_macro::{TokenStream as TokenStream1};
use proc_macro2::{Span, TokenStream};
use quote::quote;

#[proc_macro_derive(Serialize)]
pub fn npy_serialize(input: TokenStream1) -> TokenStream1 {
    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let expanded = impl_npy_serialize(&ast);

    // Return the generated impl
    expanded.into()
}

#[proc_macro_derive(Deserialize)]
pub fn npy_deserialize(input: TokenStream1) -> TokenStream1 {
    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let expanded = impl_npy_deserialize(&ast);

    // Return the generated impl
    expanded.into()
}

#[proc_macro_derive(AutoSerialize)]
pub fn npy_auto_serialize(input: TokenStream1) -> TokenStream1 {
    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let expanded = impl_npy_auto_serialize(&ast);

    // Return the generated impl
    expanded.into()
}

struct FieldData {
    idents: Vec<syn::Ident>,
    idents_str: Vec<String>,
    types: Vec<TokenStream>,
}

impl FieldData {
    fn extract(ast: &syn::DeriveInput) -> Self {
        let fields = match ast.data {
            syn::Data::Struct(ref data) => &data.fields,
            _ => panic!("npyz derive macros can only be used with structs"),
        };

        let idents: Vec<syn::Ident> = fields.iter().map(|f| {
            f.ident.clone().expect("Tuple structs not supported")
        }).collect();
        let idents_str = idents.iter().map(|t| unraw(t)).collect::<Vec<_>>();

        let types: Vec<TokenStream> = fields.iter().map(|f| {
            let ty = &f.ty;
            quote!( #ty )
        }).collect::<Vec<_>>();

        FieldData { idents, idents_str, types }
    }
}

fn impl_npy_serialize(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let vis = &ast.vis;
    let FieldData { ref idents, ref idents_str, ref types } = FieldData::extract(ast);

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let field_dtypes_struct = gen_field_dtypes_struct(idents, idents_str);

    let idents_1 = idents;

    wrap_in_const("Serialize", &name, quote! {
        use ::std::io;

        #vis struct GeneratedWriter #ty_generics #where_clause {
            writers: FieldWriters #ty_generics,
        }

        struct FieldWriters #ty_generics #where_clause {
            #( #idents: <#types as _npyz::Serialize>::TypeWriter ,)*
        }

        #field_dtypes_struct

        impl #impl_generics _npyz::TypeWrite for GeneratedWriter #ty_generics #where_clause {
            type Value = #name #ty_generics;

            #[allow(unused_mut)]
            fn write_one<W: io::Write>(&self, mut w: W, value: &Self::Value) -> io::Result<()> {
                #({ // braces for pre-NLL
                    let method = <<#types as _npyz::Serialize>::TypeWriter as _npyz::TypeWrite>::write_one;
                    method(&self.writers.#idents, &mut w, &value.#idents_1)?;
                })*
                p::Ok(())
            }
        }

        impl #impl_generics _npyz::Serialize for #name #ty_generics #where_clause {
            type TypeWriter = GeneratedWriter #ty_generics;

            fn writer(dtype: &_npyz::DType) -> p::Result<GeneratedWriter, _npyz::DTypeError> {
                let dtypes = FieldDTypes::extract(dtype)?;
                let writers = FieldWriters {
                    #( #idents: <#types as _npyz::Serialize>::writer(&dtypes.#idents_1)? ,)*
                };

                p::Ok(GeneratedWriter { writers })
            }
        }
    })
}

fn impl_npy_deserialize(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let vis = &ast.vis;
    let FieldData { ref idents, ref idents_str, ref types } = FieldData::extract(ast);

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let field_dtypes_struct = gen_field_dtypes_struct(idents, idents_str);

    let idents_1 = idents;

    wrap_in_const("Deserialize", &name, quote! {
        use ::std::io;

        #vis struct GeneratedReader #ty_generics #where_clause {
            readers: FieldReaders #ty_generics,
        }

        struct FieldReaders #ty_generics #where_clause {
            #( #idents: <#types as _npyz::Deserialize>::TypeReader ,)*
        }

        #field_dtypes_struct

        impl #impl_generics _npyz::TypeRead for GeneratedReader #ty_generics #where_clause {
            type Value = #name #ty_generics;

            #[allow(unused_mut)]
            fn read_one<R: io::Read>(&self, mut reader: R) -> io::Result<Self::Value> {
                #(
                    let func = <<#types as _npyz::Deserialize>::TypeReader as _npyz::TypeRead>::read_one;
                    let #idents = func(&self.readers.#idents_1, &mut reader)?;
                )*
                io::Result::Ok(#name { #( #idents ),* })
            }
        }

        impl #impl_generics _npyz::Deserialize for #name #ty_generics #where_clause {
            type TypeReader = GeneratedReader #ty_generics;

            fn reader(dtype: &_npyz::DType) -> p::Result<GeneratedReader, _npyz::DTypeError> {
                let dtypes = FieldDTypes::extract(dtype)?;
                let readers = FieldReaders {
                    #( #idents: <#types as _npyz::Deserialize>::reader(&dtypes.#idents_1)? ,)*
                };

                p::Ok(GeneratedReader { readers })
            }
        }
    })
}

fn impl_npy_auto_serialize(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let FieldData { idents: _, ref idents_str, ref types } = FieldData::extract(ast);

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    wrap_in_const("AutoSerialize", &name, quote! {
        impl #impl_generics _npyz::AutoSerialize for #name #ty_generics #where_clause {
            fn default_dtype() -> _npyz::DType {
                _npyz::DType::Record(::std::vec![#(
                    _npyz::Field {
                        name: p::ToString::to_string(#idents_str),
                        dtype: <#types as _npyz::AutoSerialize>::default_dtype()
                    }
                ),*])
            }
        }
    })
}

fn gen_field_dtypes_struct(
    idents: &[syn::Ident],
    idents_str: &[String],
) -> TokenStream {
    assert_eq!(idents.len(), idents_str.len());
    quote!{
        struct FieldDTypes {
            #( #idents : _npyz::DType ,)*
        }

        impl FieldDTypes {
            fn extract(dtype: &_npyz::DType) -> p::Result<Self, _npyz::DTypeError> {
                let fields = match dtype {
                    _npyz::DType::Record(fields) => fields,
                    ty => return p::Err(_npyz::DTypeError::expected_record(ty)),
                };

                let correct_names: &[&str] = &[ #(#idents_str),* ];

                if p::Iterator::ne(
                    p::Iterator::map(fields.iter(), |f| &f.name[..]),
                    p::Iterator::cloned(correct_names.iter()),
                ) {
                    let actual_names = p::Iterator::map(fields.iter(), |f| &f.name[..]);
                    return p::Err(_npyz::DTypeError::wrong_fields(actual_names, correct_names));
                }

                #[allow(unused_mut)]
                let mut fields = p::IntoIterator::into_iter(fields);
                p::Result::Ok(FieldDTypes {
                    #( #idents : {
                        let field = p::Iterator::next(&mut fields).unwrap();
                        p::Clone::clone(&field.dtype)
                    },)*
                })
            }
        }
    }
}

// from the wonderful folks working on serde.
// By placing our generated impls inside a `const`, we can freely use `use`
// and `extern crate` without them leaking into the module.
fn wrap_in_const(
    trait_: &str,
    ty: &syn::Ident,
    code: TokenStream,
) -> TokenStream {
    let dummy_const = syn::Ident::new(
        &format!("__IMPL_npy_{}_FOR_{}", trait_, unraw(ty)),
        Span::call_site(),
    );

    quote! {
        #[allow(non_upper_case_globals)]
        #[allow(unused_attributes)]
        #[allow(unused_qualifications)]
        #[allow(non_local_definitions)]  // this warns on the impl-in-a-const technique, lol
        const #dummy_const: () = {
            #[allow(unknown_lints)]
            #[clippy::allow(useless_attribute)]
            #[allow(rust_2018_idioms)]
            extern crate npyz as _npyz;

            // if our generated code directly imports any traits, then the #[no_implicit_prelude]
            // test won't catch accidental use of method syntax on trait methods (which can fail
            // due to ambiguity with similarly-named methods on other traits).  So if we want to
            // abbreviate paths, we need to do this instead:
            use ::std::prelude::v1 as p;

            #code
        };
    }
}

fn unraw(ident: &syn::Ident) -> String {
    ident.to_string().trim_start_matches("r#").to_owned()
}
