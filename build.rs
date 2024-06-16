fn main() {
    // Need this for CoreML. See: https://ort.pyke.io/perf/execution-providers#coreml
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");

    #[cfg(target_os = "linux")]
    // print rustflags = [ "-Clink-args=-Wl,-rpath,\\$ORIGIN" ]
    println!("cargo:rustc-link-arg=-Wl,-rpath,\\$ORIGIN");
}
