fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("app-icon.ico");
        res.set("FileDescription", "StealCode Update Helper");
        res.set("ProductName", "StealCode");
        if let Err(e) = res.compile() {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
