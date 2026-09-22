# Mobile builds for StealCode (Android + iOS).
#
# POSIX sh throughout (`set shell` forces sh): the same file works on
# Windows (MSYS/Git Bash sh), Linux and macOS. gradle runs through the
# POSIX `./gradlew` wrapper everywhere (see `gradlew` below); no cmd.exe,
# which under MSYS mangles `/c` into a path and never runs gradlew.bat.
#
# Android SDK comes from ANDROID_HOME (or ANDROID_SDK_ROOT); recipes that need
# it fail with an error when neither is set. No hardcoded SDK paths.
#
# Android ABI: defaults to arm64-v8a; override per build with
#   just abi=x86_64 android-release
# (one of arm64-v8a, armeabi-v7a, x86, x86_64). Gradle writes the APK to
# target/android/outputs/apk/<variant>/app-<variant>.apk, then the apk-debug /
# apk-release recipes rename it to the final path:
#   target/android/outputs/apk/<variant>/stealcode.apk
#
# iOS needs macOS + Xcode (xcodegen, xcodebuild, simctl/devicectl); ios-sim and
# ios-device refuse to run anywhere else. The Xcode project lives in
# crates/mobile/assets/ios and builds all output into target/.

set shell := ["sh", "-cu"]

android_home := env("ANDROID_HOME", env("ANDROID_SDK_ROOT", ""))
gradle_dir := "crates/mobile/assets/android/gradle"
jni_dir := "target/jniLibs"
abi := "arm64-v8a"
ndk_platform := "31"
android_targets := "aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android"
package := "com.hethinks.stealcode"
activity := "dev.gpui.mobile.GpuiActivity"
app_debug := "target/android/outputs/apk/debug/app-debug.apk"
app_release := "target/android/outputs/apk/release/app-release.apk"
apk_debug := "target/android/outputs/apk/debug/stealcode.apk"
apk_release := "target/android/outputs/apk/release/stealcode.apk"
gradlew := "./gradlew"
ios_dir := "crates/mobile/assets/ios"
ios_project := ios_dir + "/StealCode.xcodeproj"
ios_scheme := "StealCode"
ios_derived_data := "target/ios"
ios_target_sim := "aarch64-apple-ios-sim"
ios_target_device := "aarch64-apple-ios"
ios_sim_destination := "platform=iOS Simulator,OS=18.6,name=iPhone 16 Pro"

# List all recipes.
default:
    @just --list

# Fail unless the Android SDK is configured.
[private]
_sdk:
    {{ if android_home == "" { error("ANDROID_HOME or ANDROID_SDK_ROOT must be set") } else { "" } }}

# Check prerequisites: cargo-ndk + rust targets.
check:
    command -v cargo-ndk >/dev/null 2>&1 || exit 1
    rustup target add {{ android_targets }}

# Build the Rust cdylib (debug) for the ABI in the `abi` variable (default arm64-v8a). Stale jniLibs are removed;
# only libmobile.so is kept (cargo-ndk copies every cdylib it rebuilt).
rust-debug:
    rm -rf "{{ jni_dir }}"
    cargo ndk -t {{ abi }} -o {{ jni_dir }} --platform {{ ndk_platform }} build -p mobile || exit 1
    for f in "{{ jni_dir }}/{{ abi }}"/*.so; do [ "$(basename "$f")" = "libmobile.so" ] || rm -f "$f"; done
    ls -l "{{ jni_dir }}/{{ abi }}"

# Same, release: ABI from the `abi` variable (default arm64-v8a).
rust-release:
    rm -rf "{{ jni_dir }}"
    cargo ndk -t {{ abi }} -o {{ jni_dir }} --platform {{ ndk_platform }} build -p mobile --release || exit 1
    for f in "{{ jni_dir }}/{{ abi }}"/*.so; do [ "$(basename "$f")" = "libmobile.so" ] || rm -f "$f"; done
    ls -l "{{ jni_dir }}/{{ abi }}"

# Assemble the debug APK (needs jniLibs from rust-debug first).
apk-debug: _sdk
    (cd "{{ gradle_dir }}" && {{ gradlew }} -Pabis={{ abi }} assembleDebug) || exit 1
    test -f "{{ app_debug }}" || exit 1
    mv -f "{{ app_debug }}" "{{ apk_debug }}"
    ls -lh "{{ apk_debug }}"
    ls -l "{{ jni_dir }}/{{ abi }}"/*.so || true

# Same, release (signed with the release key when configured via
# keystore.properties / ANDROID_KEYSTORE_* env, otherwise the debug key;
# see app/build.gradle.kts).
apk-release: _sdk
    (cd "{{ gradle_dir }}" && {{ gradlew }} -Pabis={{ abi }} assembleRelease) || exit 1
    test -f "{{ app_release }}" || exit 1
    mv -f "{{ app_release }}" "{{ apk_release }}"
    ls -lh "{{ apk_release }}"
    ls -l "{{ jni_dir }}/{{ abi }}"/*.so || true

# Rust + APK, debug.
android-debug: rust-debug apk-debug

# Rust + APK, release.
android-release: rust-release apk-release

# Install the APK (`just install` / `just install release`).
# Retries via uninstall on debug/release signature mismatch.
install profile="debug": _sdk
    {{ if profile =~ "^(debug|release)$" { "" } else { error("unknown profile (want debug|release)") } }}
    command -v adb >/dev/null 2>&1 || exit 1
    adb devices | grep -v "List of" | grep -q "device" || exit 1
    adb install -r "{{ if profile == "release" { apk_release } else { apk_debug } }}" || { adb uninstall {{ package }}; adb install "{{ if profile == "release" { apk_release } else { apk_debug } }}"; } || exit 1

# Launch the app on the connected device/emulator.
run: _sdk
    command -v adb >/dev/null 2>&1 || exit 1
    adb shell am start -n "{{ package }}/{{ activity }}" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER

# Full pipeline: build + install + run (`just android` / `just android release`).
android profile="debug":
    {{ if profile =~ "^(debug|release)$" { "" } else { error("unknown profile (want debug|release)") } }}
    just android-{{ profile }}
    just install {{ profile }}
    just run

# Clean rust + gradle artifacts.
clean:
    for t in {{ android_targets }}; do cargo clean --target "$t"; done
    (cd "{{ gradle_dir }}" && {{ gradlew }} clean) || exit 1

# Build the Rust staticlib, generate the Xcode project, build it and run on
# the simulator (macOS + Xcode only).
ios-sim profile="debug":
    {{ if os() != "macos" { error("iOS builds require macOS and Xcode") } else { "" } }}
    {{ if profile =~ "^(debug|release)$" { "" } else { error("unknown profile (want debug|release)") } }}
    rustup target add {{ ios_target_sim }}
    cargo build -p mobile --target {{ ios_target_sim }} {{ if profile == "release" { "--release" } else { "" } }}
    xcodegen generate --spec {{ ios_dir }}/project.yml
    xcodebuild -project {{ ios_project }} -scheme {{ ios_scheme }} -configuration {{ if profile == "release" { "Release" } else { "Debug" } }} -destination "{{ ios_sim_destination }}" -derivedDataPath {{ ios_derived_data }} build
    sim_id="$(xcrun simctl list devices available | grep "iPhone" | head -1 | sed -E 's/.*\(([A-F0-9-]+)\).*/\1/')"; test -n "$sim_id" || exit 1
    xcrun simctl boot "$sim_id" 2>/dev/null || true
    xcrun simctl bootstatus "$sim_id" -b
    app="$(find {{ ios_derived_data }} -path "*/Build/Products/{{ if profile == "release" { "Release" } else { "Debug" } }}-iphonesimulator/{{ ios_scheme }}.app" -type d | head -1)"; test -n "$app" || exit 1
    xcrun simctl install "$sim_id" "$app"
    xcrun simctl launch --terminate-running-process "$sim_id" {{ package }}

# Same, on a connected device via devicectl (macOS + Xcode only).
ios-device profile="debug":
    {{ if os() != "macos" { error("iOS builds require macOS and Xcode") } else { "" } }}
    {{ if profile =~ "^(debug|release)$" { "" } else { error("unknown profile (want debug|release)") } }}
    rustup target add {{ ios_target_device }}
    cargo build -p mobile --target {{ ios_target_device }} {{ if profile == "release" { "--release" } else { "" } }}
    xcodegen generate --spec {{ ios_dir }}/project.yml
    xcodebuild -project {{ ios_project }} -scheme {{ ios_scheme }} -configuration {{ if profile == "release" { "Release" } else { "Debug" } }} -destination "generic/platform=iOS" -derivedDataPath {{ ios_derived_data }} build
    app="$(find {{ ios_derived_data }} -path "*/Build/Products/{{ if profile == "release" { "Release" } else { "Debug" } }}-iphoneos/{{ ios_scheme }}.app" -type d | head -1)"; test -n "$app" || exit 1
    device_id="$(xcodebuild -project {{ ios_project }} -scheme {{ ios_scheme }} -showdestinations 2>/dev/null | grep "platform:iOS," | grep -v Simulator | grep -v placeholder | head -1 | sed -E 's/.*id:([^,}]+).*/\1/' | tr -d '[:space:]')"; test -n "$device_id" || exit 1
    xcrun devicectl device install app --device "$device_id" "$app"
    xcrun devicectl device process launch --device "$device_id" {{ package }}
