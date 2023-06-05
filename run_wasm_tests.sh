#!/bin/bash

set -e
set -x

wasm-pack test --node --test nd
wasm-pack test --node --test roundtrip --features derive
wasm-pack test --node --test serialize_array --features derive
