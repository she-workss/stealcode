//! Binary stub for the `StealCode` Android crate.
//!
//! On **Android** the `android-activity` crate invokes `android_main` directly
//! from the cdylib defined in `lib.rs` - so `main` is never called. We still
//! provide a stub so that `cargo check` (without a target triple) succeeds.

#[cfg(target_os = "android")]
fn main() {
    // On Android the real entry point is `android_main` in lib.rs, which is
    // called by the `android-activity` crate from the cdylib. This binary
    // target is unused but must compile.
    eprintln!(
        "This binary is not used on Android. The app enters via android_main() in lib.rs."
    );
}

#[cfg(not(target_os = "android"))]
fn main() {
    // Allow `cargo check` / `cargo clippy` on the host to succeed without
    // requiring a mobile target.
    eprintln!(
        "This crate is designed for Android. Please build via cargo-ndk (Android)."
    );
}
