use rustc_version::{Channel, version_meta};

fn main() {
    // Check if the "simd" feature is enabled
    let simd_enabled = std::env::var_os("CARGO_FEATURE_SIMD").is_some();

    // Check the current compiler channel
    let channel = version_meta().unwrap().channel;

    if simd_enabled && channel != Channel::Nightly {
        panic!(
            "\n\nError: The 'simd' feature requires a nightly compiler.\n\
             Please use 'rustup run nightly cargo build --features simd' or set 'rustup default nightly'.\n"
        );
    }

    // Optional: Emit a custom cfg flag if you want to use it in your code
    if channel == Channel::Nightly {
        println!("cargo:rustc-cfg=nightly");
    }
}
