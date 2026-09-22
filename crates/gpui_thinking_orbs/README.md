# gpui-thinking-orbs

https://github.com/user-attachments/assets/a855216d-f07a-47de-804d-f096a80d9d7a

Dotted thought-orb loading indicators for AI & agent UIs - a native
**[GPUI](https://www.gpui.rs/)** library port of
[thinking-orbs](https://orbs.jakubantalik.com/)
([Jakub Antalik](https://github.com/Jakubantalik/thinking-orbs), MIT).

Twelve hand-tuned animated states, four size presets, monochrome light/dark ink.
Geometry is pure Rust; painting uses GPUI `canvas` + rounded quads / stroked paths.
No WebGL, no browser, no React.

### Status / roadmap

This is early **0.x**. There is still **a lot to improve**: motion polish, density
tuning per size, new states, and host-integration ergonomics. New orbs will keep
landing, and existing ones will keep getting refined - expect visible iteration
rather than a frozen gallery.

---

## Usage

```rust
use gpui_thinking_orbs::{OrbState, OrbSize, OrbTheme, ThinkingOrb};

// As a window root, or nested via Entity:
cx.new(|_| {
    ThinkingOrb::new()
        .state(OrbState::Searching)
        .size(OrbSize::Avatar)
        .theme(OrbTheme::Auto)
        .speed(1.0)
})
```

### States

| State | Animation |
|-------|-----------|
| `Working` | particles on tilted orbits |
| `Searching` | scan meridian sweeps a dotted globe |
| `Solving` | bands scramble, then click back solved |
| `Listening` | waveform rolls through latitude rings |
| `Connecting` | constellation wires itself + packets |
| `Weaving` | three strands plait around the sphere |
| `Composing` | undulating multi-band sash |
| `Breathing` | face-on ring slowly morphing |
| `Shaping` | dotted outline: circle → triangle → square |
| `Focusing` | particle iris converges on a focal core |
| `Reasoning` | counter-rotating gyroscope loops |
| `Recalling` | memory echoes expand from a steady core |

### Sizes

Four tuned designs:

- `OrbSize::Inline` - 20 px (inline text)
- `OrbSize::Avatar` - 64 px (chat avatar)
- `OrbSize::Large` - 96 px (prominent card/status)
- `OrbSize::Hero` - 128 px (hero or empty state)

Large and Hero preserve the Avatar design while progressively increasing dot
density and radius; they are not merely a stretched 64 px canvas. Iterate all
available sizes with `OrbSize::ALL_SIZES`.

### Theme

| Value | Meaning |
|-------|---------|
| `OrbTheme::Auto` | Follow `WindowAppearance` (updates on OS light/dark change) |
| `OrbTheme::Dark` | Light ink (for dark backgrounds) |
| `OrbTheme::Light` | Dark ink (for light backgrounds) |

### Reduced motion & visibility

```rust
ThinkingOrb::new()
    .reduced_motion(true) // static frame at t = 0.6, no timer
    .visible(true) // set false when the entity is mounted but off-screen
```

GPUI has no intersection observer. If a list row recycles an orb that scrolled
away, call `set_visible(false)` or unmount the entity so it stops ticking.

---

## Performance

Built to stay cheap in real agent UIs - chat sidebars, status chips, many orbs
on screen - not as a full-screen particle toy.

### Power-user paint loop

```rust
use gpui_thinking_orbs::{draw_mode, resolve_preset, OrbState, OrbSize};

let resolved = resolve_preset(OrbState::Working, OrbSize::Avatar);
let frame = draw_mode(resolved.mode, 64.0, t_seconds, &resolved.opts);
// frame.dots / frame.lines - paint however you like
```

---

## License

MIT. The original nine animation designs and tuning are © Jakub Antalik
(thinking-orbs). `Focusing`, `Reasoning`, and `Recalling` are original concepts
for this GPUI library. This port reimplements the engine for native Rust apps.
