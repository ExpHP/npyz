#![recursion_limit = "256"]

/*!
Derive `trait Serializable` for a structure.

Using this crate, it is enough to `#[derive(npy::Serialize, npy::Deserialize)]` on a struct to be able to
serialize and deserialize it. All of the fields must implement [`Serialize`](../npy/trait.Serialize.html)
and [`Deserialize`](../npy/trait.Deserialize.html) respectively.

*/

extern crate proc_macro;
extern crate proc_macro2;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::Tokens;

/// Macros 1.1-based custom derive function
#[proc_macro_derive(Serialize)]
pub fn npy_serialize(input: TokenStream) -> TokenStream {
    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let expanded = impl_npy_serialize(&ast);

    // Return the generated impl
    expanded.into()
}

#[proc_macro_derive(Deserialize)]
pub fn npy_deserialize(input: TokenStream) -> TokenStream {
    // Parse the string representation
    let ast = syn::parse(input).unwrap();

    // Build the impl
    let expanded = impl_npy_deserialize(&ast);

    // Return the generated impl
    expanded.into()
}

#[proc_macro_derive(AutoSerialize)]
pub fn npy_auto_serialize(input: TokenStream) -> TokenStream {
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
    types: Vec<Tokens>,
}

impl FieldData {
    fn extract(ast: &syn::DeriveInput) -> Self {
        let fields = match ast.data {
            syn::Data::Struct(ref data) => &data.fields,
            _ => panic!("npy derive macros can only be used with structs"),
        };

        let idents: Vec<syn::Ident> = fields.iter().map(|f| {
            f.ident.clone().expect("Tuple structs not supported")
        }).collect();
        let idents_str = idents.iter().map(|t| unraw(t)).collect::<Vec<_>>();

        let types: Vec<Tokens> = fields.iter().map(|f| {
            let ty = &f.ty;
            quote!( #ty )
        }).collect::<Vec<_>>();

        FieldData { idents, idents_str, types }
    }
}

fn impl_npy_serialize(ast: &syn::DeriveInput) -> Tokens {
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
            #( #idents: <#types as _npy::Serialize>::Writer ,)*
        }

        #field_dtypes_struct

        impl #impl_generics _npy::TypeWrite for GeneratedWriter #ty_generics #where_clause {
            type Value = #name #ty_generics;

            #[allow(unused_mut)]
            fn write_one<W: io::Write>(&self, mut w: W, value: &Self::Value) -> io::Result<()> {
                #(
                    let method = <<#types as _npy::Serialize>::Writer as _npy::TypeWrite>::write_one;
                    method(&self.writers.#idents, &mut w, &value.#idents_1)?;
                )*
                p::Ok(())
            }
        }

        impl #impl_generics _npy::Serialize for #name #ty_generics #where_clause {
            type Writer = GeneratedWriter #ty_generics;

            fn writer(dtype: &_npy::DType) -> p::Result<GeneratedWriter, _npy::DTypeError> {
                let dtypes = FieldDTypes::extract(dtype)?;
                let writers = FieldWriters {
                    #( #idents: <#types as _npy::Serialize>::writer(&dtypes.#idents_1)? ,)*
                };

                p::Ok(GeneratedWriter { writers })
            }
        }
    })
}

fn impl_npy_deserialize(ast: &syn::DeriveInput) -> Tokens {
    let name = &ast.ident;
    let vis = &ast.vis;
    let FieldData { ref idents, ref idents_str, ref types } = FieldData::extract(ast);

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let field_dtypes_struct = gen_field_dtypes_struct(idents, idents_str);

    let idents_1 = idents;

    wrap_in_const("Deserialize", &name, quote! {
        #vis struct GeneratedReader #ty_generics #where_clause {
            readers: FieldReaders #ty_generics,
        }

        struct FieldReaders #ty_generics #where_clause {
            #( #idents: <#types as _npy::Deserialize>::Reader ,)*
        }

        #field_dtypes_struct

        impl #impl_generics _npy::TypeRead for GeneratedReader #ty_generics #where_clause {
            type Value = #name #ty_generics;

            #[allow(unused_mut)]
            fn read_one<'a>(&self, mut remainder: &'a [u8]) -> (Self::Value, &'a [u8]) {
                #(
                    let func = <<#types as _npy::Deserialize>::Reader as _npy::TypeRead>::read_one;
                    let (#idents, new_remainder) = func(&self.readers.#idents_1, remainder);
                    remainder = new_remainder;
                )*
                (#name { #( #idents ),* }, remainder)
            }
        }

        impl #impl_generics _npy::Deserialize for #name #ty_generics #where_clause {
            type Reader = GeneratedReader #ty_generics;

            fn reader(dtype: &_npy::DType) -> p::Result<GeneratedReader, _npy::DTypeError> {
                let dtypes = FieldDTypes::extract(dtype)?;
                let readers = FieldReaders {
                    #( #idents: <#types as _npy::Deserialize>::reader(&dtypes.#idents_1)? ,)*
                };

                p::Ok(GeneratedReader { readers })
            }
        }
    })
}

fn impl_npy_auto_serialize(ast: &syn::DeriveInput) -> Tokens {
    let name = &ast.ident;
    let FieldData { idents: _, ref idents_str, ref types } = FieldData::extract(ast);

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    wrap_in_const("AutoSerialize", &name, quote! {
        impl #impl_generics _npy::AutoSerialize for #name #ty_generics #where_clause {
            fn default_dtype() -> _npy::DType {
                _npy::DType::Record(vec![#(
                    _npy::Field {
                        name: #idents_str.to_string(),
                        dtype: <#types as _npy::AutoSerialize>::default_dtype()
                    }
                ),*])
            }
        }
    })
}

fn gen_field_dtypes_struct(
    idents: &[syn::Ident],
    idents_str: &[String],
) -> Tokens {
    assert_eq!(idents.len(), idents_str.len());
    quote!{
        struct FieldDTypes {
            #( #idents : _npy::DType ,)*
        }

        impl FieldDTypes {
            fn extract(dtype: &_npy::DType) -> p::Result<Self, _npy::DTypeError> {
                let fields = match dtype {
                    _npy::DType::Record(fields) => fields,
                    _npy::DType::Plain { ty, .. } => return p::Err(_npy::DTypeError::expected_record(ty)),
                };

                let correct_names: &[&str] = &[ #(#idents_str),* ];

                if p::Iterator::ne(
                    p::Iterator::map(fields.iter(), |f| &f.name[..]),
                    p::Iterator::cloned(correct_names.iter()),
                ) {
                    let actual_names = p::Iterator::map(fields.iter(), |f| &f.name[..]);
                    return p::Err(_npy::DTypeError::wrong_fields(actual_names, correct_names));
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

// from the wonderful folks working on serde
fn wrap_in_const(
    trait_: &str,
    ty: &syn::Ident,
    code: Tokens,
) -> Tokens {
    let dummy_const = syn::Ident::new(
        &format!("__IMPL_npy_{}_FOR_{}", trait_, unraw(ty)),
        Span::call_site(),
    );

    quote! {
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const #dummy_const: () = {
            #[allow(unknown_lints)]
            #[cfg_attr(feature = "cargo-clippy", allow(useless_attribute))]
            #[allow(rust_2018_idioms)]
            extern crate npy as _npy;

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
