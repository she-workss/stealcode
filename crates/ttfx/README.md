# ttfx

Terminal text effects as a single static binary. Pipe text in, pick an effect:

```sh
ls -la | ttfx decrypt
cat banner.txt | ttfx beams
fortune | ttfx --random-effect
git log --oneline -10 | ttfx matrix
```

## Credit where it's due

**This is a port of [TerminalTextEffects](https://github.com/ChrisBuilds/terminaltexteffects)
(TTE) by [ChrisBuilds](https://github.com/ChrisBuilds).** Every effect, the animation engine,
and the command-line interface are their design - this project translates that work to Rust
and adds nothing of its own to the art. If you like what you see here, star the original.

TTE is MIT licensed and so is this port; the original copyright is preserved in
[LICENSE](LICENSE) and [NOTICE](NOTICE). Please file *effect* ideas upstream, where they belong.

## Why a port

TTE is a Python package. That's the right call for a library, but for a shell toy that lives in
your prompt pipeline it means an interpreter, an install step, and ~65 ms of import before the
first frame. ttfx is one dependency-free binary that starts in half a millisecond.

That difference is the whole reason this exists. On a fullscreen canvas the heavier effects run
out of headroom under Python. Time to render a whole animation, pacing disabled so this measures
throughput rather than `sleep()`:

| At 200×50 cells | frames | ttfx | Python TTE | ttfx fps |
|---|---|---|---|---|
| slide | 375 | 76 ms | 2,203 ms | 4,930 |
| beams | 732 | 181 ms | 5,564 ms | 4,050 |
| waves | 633 | 374 ms | 8,745 ms | 1,693 |
| startup | - | 0.5 ms | 64 ms | - |

Across the 35 effects that aren't gated on wall-clock time, the median speedup is **27.5×**
(range 17.1×-47.4×). The two that are gated - `matrix` and `thunderstorm` - spend most of their
runtime in a fixed animation duration that no implementation can shorten, so they come in at
1.9× and 1.3×; what ttfx buys there is a far higher frame rate inside that window, not a shorter
one.

Reproduce it with `python3 tools/tests/bench_full.py`, or set `TTFX_BENCH_COLS`, `TTFX_BENCH_LINES`
and `TTFX_BENCH_FILL=1` for the fullscreen numbers above. Both sides run their real user-facing
command, best of five.

## Fidelity

This is a *parity port*, not a reimplementation-in-spirit. Given the same input, config, and
random draws, ttfx produces **byte-identical frames** to the Python original - verified
mechanically in CI against a pinned upstream checkout (v0.15.0), not by eyeballing.

| Suite | Checks | What it proves |
|---|---|---|
| `tools/parity/run_suite.sh` | 354 | every effect's frame stream, byte for byte, across configs and seeds |
| `tools/parity/tty_compare.sh` | 41 | the full terminal byte stream - canvas prep, cursor moves, teardown |
| `tools/tests/cli_corpus.sh` | 19 | exit codes and stdout/stderr routing |
| `tools/tests/*_behavior.py` | pty | what only a real terminal shows: resize restarts, signal teardown |
| `cargo test` | goldens + traces | easing/geometry/gradient values and engine state machines |

`./bin/test` runs the lot, which is all CI does.

Making that possible meant reproducing upstream's quirks deliberately, not "fixing" them:
Python's banker's rounding, gradients built from integer floor division rather than float
interpolation, a bezier arc-length approximation that drops its final segment, and looping
scenes that report themselves complete on every tick. They're catalogued in
[`plan.md`](plan.md).

**Two deliberate differences.** Random number generation is not bit-compatible with CPython -
ttfx uses xoshiro256++, so `--seed` is reproducible within ttfx but won't match Python's
Mersenne Twister. (The parity harness swaps a shared PRNG into both sides, which is what makes
frame comparison possible at all.) And Python plugin effects aren't supported, since there's
no interpreter to load them.

## Usage

```
<producer> | ttfx [terminal options] <effect> [effect options]

ttfx --help                 # all 33 effects and the terminal options
ttfx <effect> --help        # options for one effect
ttfx --random-effect        # surprise me (--include-effects / --exclude-effects to filter)
ttfx --print-completion bash|zsh
```

Terminal options (canvas size and anchoring, color handling, frame rate, text wrapping) go
before the effect name; effect options after it. Option names and defaults match `tte`, so
existing invocations work with the binary name swapped.

## Building

```sh
cargo build --release
cargo build --release --target x86_64-unknown-linux-musl   # static, ~3.3 MB
```

`./bin/test` runs every suite. It needs python3, and the parity half needs a copy of
upstream, which it clones at the pinned commit on first run:

```sh
./tools/parity/fetch_reference.sh   # what bin/test calls; safe to run by hand
```

Upstream is not vendored here - the harness fetches it, because it's their code.

## Scope

Linux and macOS. Built for [Omarchy](https://omarchy.org) originally; nothing targets a
specific libc, and CI runs the tests and CLI corpus on both platforms. The byte-exact
parity suites stay pinned to Linux/glibc - Apple's libm rounds a few transcendentals a
last-ulp differently, which quantization hides in real frames but a bit-exact comparison
would surface.

## License

MIT - see [LICENSE](LICENSE), which carries both this project's copyright and the original
TerminalTextEffects copyright, and [NOTICE](NOTICE) for the attribution in full.
