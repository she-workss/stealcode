use std::path::Path;

fn main() {
    let windows_icon = Path::new("assets/icons/prod/icon.ico");
    let macos_plist = Path::new("assets/resources/macos/Info.plist");
    let macos_icon = Path::new("assets/icons/prod/icon.icns");
    let linux_desktop = Path::new("assets/resources/linux/stealcode.desktop");
    let linux_icon = Path::new("assets/icons/prod/icon.png");
    println!("cargo:rerun-if-changed={}", windows_icon.display());
    println!("cargo:rerun-if-changed={}", macos_plist.display());
    println!("cargo:rerun-if-changed={}", macos_icon.display());
    println!("cargo:rerun-if-changed={}", linux_desktop.display());
    println!("cargo:rerun-if-changed={}", linux_icon.display());
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        if let Ok(explicit_rc_toolkit_path) =
            std::env::var("STEALCODE_RC_TOOLKIT_PATH")
        {
            res.set_toolkit_path(explicit_rc_toolkit_path.as_str());
        }
        res.set_icon(windows_icon.to_str().expect("invalid icon path"));
        res.set("FileDescription", "StealCode");
        res.set("ProductName", "StealCode");
        res.set("OriginalFilename", "stealcode.exe");
        res.set("LegalCopyright", "Copyright © he-thinks 2026");
        if let Err(e) = res.compile() {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
    #[cfg(target_os = "macos")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let out_plist = Path::new(&out_dir).join("Info.plist");
        std::fs::copy(macos_plist, &out_plist)
            .expect("Failed to copy Info.plist");
        println!(
            "cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{}",
            out_plist.display()
        );
    }
}
