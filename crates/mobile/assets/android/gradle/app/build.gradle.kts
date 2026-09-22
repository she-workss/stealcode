import java.io.File

// App module build.gradle.kts for StealCode Android.
//
// This module packages the pre-compiled Rust native library into an APK
// that uses Android's NativeActivity to host the GPUI application.
//
// The Rust library must be compiled separately into target/jniLibs/<abi>/
// before building the APK. Gradle writes the APK to
// target/android/outputs/apk/<variant>/app-<variant>.apk; the justfile
// (apk-debug / apk-release recipes) renames it to
// target/android/outputs/apk/<variant>/stealcode.apk and passes the target
// ABI via -Pabis=<abi>.
//
// Version: versionCode/versionName come from the workspace Cargo.toml
// ([workspace.package] version) so Android never drifts from the desktop
// binaries. 0.14.38 -> versionName "0.14.38", versionCode 14038.
//
// Release signing: locally create a gitignored `keystore.properties` in the
// repo root with these keys. It is parsed as plain `key=value` lines (NOT
// java.util.Properties), so backslashes are NOT escapes and Windows paths
// work as written, e.g. `storeFile=D:\keys\stealcode.jks` or
// `storeFile=stealcode-release.jks` (relative to the repo root). Lines whose
// first non-space character is `#` are comments; `#` inside a value is kept
// verbatim (passwords may contain `#`):
//   storeFile=<path to .jks/.keystore, e.g. D:\keys\stealcode.jks>
//   storePassword=<...>
//   keyAlias=<...>
//   keyPassword=<...>
// In CI the same four values come from ANDROID_KEYSTORE_PATH /
// ANDROID_KEYSTORE_PASSWORD / ANDROID_KEY_ALIAS / ANDROID_KEY_PASSWORD (the
// workflow decodes the base64 keystore secret to a file first). When neither
// is configured, release builds fall back to the debug key so
// `just android-release` still produces an installable APK.
//
// Quick start:
//   cargo ndk -t arm64-v8a -o target/jniLibs \
//       --platform 31 build -p mobile --release
//   cd crates/mobile/assets/android/gradle
//   ./gradlew -Pabis=arm64-v8a assembleDebug

plugins {
    id("com.android.application")
}

// Repo root: <repo>/crates/mobile/assets/android/gradle -> <repo> (5 levels up).
val repoRoot = rootProject.projectDir.parentFile
    .parentFile.parentFile.parentFile.parentFile

// Version: the workspace Cargo.toml is the single source of truth (the
// same version the desktop binaries report).
val cargoTomlFile = repoRoot.resolve("Cargo.toml")
val workspaceSection = cargoTomlFile.readText().substringAfter("[workspace.package]", "")
require(workspaceSection.isNotEmpty()) { "no [workspace.package] section in $cargoTomlFile" }
val cargoVersion = Regex("""version\s*=\s*"([^"]+)"""").find(workspaceSection)
    ?.groupValues?.get(1)
    ?: error("no version in [workspace.package] of $cargoTomlFile")
// 0.14.38 -> 14038. Assumes minor and patch stay below 1000.
val versionParts = cargoVersion.substringBefore('-').split('.').mapNotNull { it.toIntOrNull() }
val androidVersionCode = versionParts.getOrElse(0) { 0 } * 1_000_000 +
    versionParts.getOrElse(1) { 0 } * 1_000 +
    versionParts.getOrElse(2) { 0 }

// Release signing. Locally: `keystore.properties` in the repo root
// (gitignored), parsed as plain `key=value` lines so Windows paths keep
// their backslashes. In CI: ANDROID_KEYSTORE_PATH / ANDROID_KEYSTORE_PASSWORD
// / ANDROID_KEY_ALIAS / ANDROID_KEY_PASSWORD (the workflow decodes the
// base64 keystore secret to a file first). Without either, release
// builds fall back to the debug key so `just android-release` still
// produces an installable APK locally.
val signingProps: Map<String, String> = repoRoot.resolve("keystore.properties")
    .takeIf { it.exists() }
    ?.readLines()
    ?.map { it.trim() }
    ?.filter { it.isNotEmpty() && !it.startsWith("#") && it.contains('=') }
    ?.associate { it.substringBefore('=').trim() to it.substringAfter('=').trim() }
    ?: emptyMap()
fun signingValue(name: String, env: String): String? =
    signingProps[name]?.takeIf { it.isNotBlank() }
        ?: System.getenv(env)?.takeIf { it.isNotBlank() }

val releaseStorePath = signingValue("storeFile", "ANDROID_KEYSTORE_PATH")
val releaseStorePassword = signingValue("storePassword", "ANDROID_KEYSTORE_PASSWORD")
val releaseKeyAlias = signingValue("keyAlias", "ANDROID_KEY_ALIAS")
val releaseKeyPassword = signingValue("keyPassword", "ANDROID_KEY_PASSWORD")
val hasReleaseSigning = listOf(releaseStorePath, releaseStorePassword, releaseKeyAlias, releaseKeyPassword).all { it != null }

// ABIs packaged into the APK (comma-separated). Defaults to the justfile's
// default cargo-ndk ABI; the justfile passes -Pabis=<abi> for other ABIs.
val abis = (findProperty("abis") as String?)
    ?.split(',')?.map(String::trim)?.filter(String::isNotEmpty)
    ?: listOf("arm64-v8a")

// AGP 9 removed `applicationVariants`; redirect the module's build output to
// the repo-level target/ directory. jniLibs stay a sibling (target/jniLibs),
// NOT nested under here, so `clean` never wipes the pre-built Rust library.
layout.buildDirectory.set(repoRoot.resolve("target/android"))

android {
    namespace = "com.hethinks.stealcode"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.hethinks.stealcode"
        minSdk = 26          // Vulkan 1.0 is mandatory from API 26+
        targetSdk = 34
        versionCode = androidVersionCode
        versionName = cargoVersion

        // Tell NativeActivity which .so to load.
        // This must match the cdylib output name (libmobile.so -> "mobile").
        ndk {
            abiFilters.addAll(abis)
        }

        // Forward the library name to the manifest via a placeholder.
        manifestPlaceholders["nativeLibraryName"] = "mobile"
    }

    signingConfigs {
        if (hasReleaseSigning) {
            create("release") {
                val storePath = releaseStorePath!!
                val path = File(storePath)
                storeFile = if (path.isAbsolute) path else repoRoot.resolve(storePath)
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        release {
            signingConfig = if (hasReleaseSigning) {
                signingConfigs.getByName("release")
            } else {
                logger.warn("keystore.properties / ANDROID_KEYSTORE_* not set - signing the release APK with the DEBUG key")
                signingConfigs.getByName("debug")
            }
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isDebuggable = true
            isJniDebuggable = true
        }
    }

    // We do NOT use CMake / ndk-build - the native library is compiled
    // externally via cargo-ndk and placed directly into target/jniLibs.
    //
    // Disable the built-in native build system so Gradle doesn't look for
    // a CMakeLists.txt or Android.mk.
    externalNativeBuild {
        // Intentionally left empty.
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    // Tell Gradle where the pre-built .so files live (repo-level target/).
    // AGP 9: use the `directories` mutable set; `srcDirs(...)` is deprecated.
    sourceSets {
        getByName("main") {
            jniLibs.directories.add(repoRoot.resolve("target/jniLibs").absolutePath)
        }
    }

    // The prebuilt .so files in jniLibs are stripped by Cargo in release
    // mode (strip = true) and by AGP when packaging. Do NOT add
    // keepDebugSymbols here - it disables AGP stripping and produces
    // ~1 GB APKs from unstripped debug builds that fail to install.

    // Lint configuration - relaxed for an example project.
    lint {
        abortOnError = false
        checkReleaseBuilds = false
    }
}

dependencies {
    // AndroidX SplashScreen compat (used by GpuiActivity to hold splash until native init)
    implementation("androidx.core:core-splashscreen:1.0.1")
}

// AGP's package<Variant> task rewrites an existing APK in place. A stored
// (uncompressed) entry such as the jniLib .so leaves its old bytes behind as
// an unreferenced hole when its size changes, so the APK only ever grows
// (46.6 MB .so -> 94.3 MB APK after one swap). Delete previous APK outputs
// before packaging so every assemble writes a fresh, whole archive. The
// cleanup task has no outputs, so it always runs and also forces a stale,
// UP-TO-DATE package task to rebuild instead of leaving dead bytes in place.
val cleanStaleApks by tasks.registering {
    outputs.upToDateWhen { false }
    doLast {
        layout.buildDirectory.dir("outputs/apk").get().asFileTree
            .matching { include("**/*.apk") }
            .forEach { it.delete() }
    }
}

tasks.matching { it.name == "packageDebug" || it.name == "packageRelease" }
    .configureEach { dependsOn(cleanStaleApks) }
