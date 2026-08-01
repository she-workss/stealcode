fn main() {
    println!(
        "cargo::rustc-check-cfg=cfg(__do_not_set_stealcode_release_channel)"
    );
    println!("cargo::rerun-if-env-changed=STEALCODE_RELEASE_CHANNEL");
    // Enabling this `cfg` will cause a runtime panic if
    // `STEALCODE_RELEASE_CHANNEL` is not also set at compile time. Don't set
    // the `cfg` directly - only set the env var (hence the name). This
    // exists for build systems (e.g. Nix's `crane`) that vendor and build
    // each crate in isolation, where the relative `include_str!` in
    // `src/lib.rs` can't see `crates/cli/RELEASE_CHANNEL`.
    if std::env::var("STEALCODE_RELEASE_CHANNEL").is_ok() {
        println!("cargo::rustc-cfg=__do_not_set_stealcode_release_channel");
    }
}
