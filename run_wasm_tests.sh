#!/bin/bash

set -e
set -x

wasm-pack test --node --test nd
wasm-pack test --node --test roundtrip --features derive --features half
wasm-pack test --node --test npz --features derive --features npz
wasm-pack test --node --test serialize_array --features derive
