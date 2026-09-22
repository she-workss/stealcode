// Root build.gradle.kts for the StealCode Android project.
//
// This is a minimal Gradle project that packages the Rust native library
// (compiled separately via cargo-ndk) into an APK using NativeActivity.
//
// All Gradle build output goes to the repo-level `target/` directory:
//   - APK:        target/android/outputs/apk/<variant>/app-<variant>.apk
//                 (renamed by the justfile to .../<variant>/stealcode.apk)
//   - jniLibs:    target/jniLibs/<abi>/libmobile.so
//
// Build steps:
//   1. Compile the Rust library:
//      cargo ndk -t arm64-v8a -o target/jniLibs --platform 31 build -p mobile --release
//
//   2. Build the APK:
//      ./gradlew assembleDebug
//
//   3. Install on device/emulator:
//      adb install target/android/outputs/apk/debug/stealcode.apk

// Repo root: <repo>/crates/mobile/assets/android/gradle -> <repo> (5 levels up).
val repoRoot = rootProject.projectDir.parentFile
    .parentFile.parentFile.parentFile.parentFile

buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:9.1.0")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.9.22")
    }
}

tasks.register("clean", Delete::class) {
    delete(rootProject.layout.buildDirectory)
    // App module redirects its buildDirectory here (see app/build.gradle.kts).
    delete(repoRoot.resolve("target/android"))
}

// ── Convenience task: build Rust + APK in one go ────────────────────────────

tasks.register<Exec>("buildRustRelease") {
    group = "rust"
    description = "Compile the Rust native library for arm64-v8a using cargo-ndk."
    workingDir = repoRoot
    commandLine(
        "cargo", "ndk",
        "-t", "arm64-v8a",
        "-o", "target/jniLibs",
        "--platform", "31",
        "build", "-p", "mobile", "--release"
    )
}

tasks.register<Exec>("buildRustDebug") {
    group = "rust"
    description = "Compile the Rust native library for arm64-v8a (debug) using cargo-ndk."
    workingDir = repoRoot
    commandLine(
        "cargo", "ndk",
        "-t", "arm64-v8a",
        "-o", "target/jniLibs",
        "--platform", "31",
        "build", "-p", "mobile"
    )
}

tasks.register("buildAll") {
    group = "rust"
    description = "Build Rust library (release) and then assemble the debug APK."
    dependsOn("buildRustRelease")
    finalizedBy(":app:assembleDebug")
}
