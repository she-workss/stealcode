//! Public types shared by the animation engine and the GPUI widget.

/// The twelve shipped states - each a hand-tuned animation.
///
/// Marked `#[non_exhaustive]`: matching on this from another crate needs a
/// wildcard arm, so shipping a tenth state is not a breaking change. Prefer
/// [`OrbState::label`] / [`OrbState::as_str`] over matching where you can.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OrbState {
    /// Particles on tilted orbits.
    #[default]
    Working,
    /// A scan meridian sweeps a dotted globe.
    Searching,
    /// Bands scramble in quarter turns, then click back.
    Solving,
    /// A waveform rolls through latitude rings.
    Listening,
    /// A constellation wires itself, packets running the edges.
    Connecting,
    /// Three strands plait around the sphere.
    Weaving,
    /// An undulating multi-band sash.
    Composing,
    /// A face-on ring slowly morphing.
    Breathing,
    /// A dotted outline morphs circle → triangle → square.
    Shaping,
    /// An iris of particles converges and relaxes around a focal point.
    Focusing,
    /// Counter-rotating great circles form a reasoning gyroscope.
    Reasoning,
    /// Concentric memory echoes travel out from a steady core.
    Recalling,
}

impl OrbState {
    /// The original nine-state gallery.
    ///
    /// Kept at its original array type for source compatibility. New code
    /// should iterate [`Self::ALL_STATES`].
    #[deprecated(since = "0.2.0", note = "use OrbState::ALL_STATES")]
    pub const ALL: [Self; 9] = [
        Self::Working,
        Self::Searching,
        Self::Solving,
        Self::Listening,
        Self::Connecting,
        Self::Weaving,
        Self::Composing,
        Self::Breathing,
        Self::Shaping,
    ];
    /// All states in playground / gallery order.
    ///
    /// This is a slice so adding future states does not change its public type.
    pub const ALL_STATES: &'static [Self] = &[
        Self::Working,
        Self::Searching,
        Self::Solving,
        Self::Listening,
        Self::Connecting,
        Self::Weaving,
        Self::Composing,
        Self::Breathing,
        Self::Shaping,
        Self::Focusing,
        Self::Reasoning,
        Self::Recalling,
    ];

    /// Human-readable status label (matches upstream aria defaults).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Working => "Working…",
            Self::Searching => "Searching…",
            Self::Solving => "Solving…",
            Self::Listening => "Listening…",
            Self::Connecting => "Connecting…",
            Self::Weaving => "Weaving…",
            Self::Composing => "Composing…",
            Self::Breathing => "Thinking…",
            Self::Shaping => "Shaping…",
            Self::Focusing => "Focusing…",
            Self::Reasoning => "Reasoning…",
            Self::Recalling => "Recalling…",
        }
    }

    /// Stable `snake_case` name (matches the web package `state` prop).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Searching => "searching",
            Self::Solving => "solving",
            Self::Listening => "listening",
            Self::Connecting => "connecting",
            Self::Weaving => "weaving",
            Self::Composing => "composing",
            Self::Breathing => "breathing",
            Self::Shaping => "shaping",
            Self::Focusing => "focusing",
            Self::Reasoning => "reasoning",
            Self::Recalling => "recalling",
        }
    }
}

/// Rendered size in logical pixels. Four tuned presets ship: inline (20),
/// avatar (64), large (96), and hero (128). Larger sizes add detail gradually
/// instead of merely stretching the 64 px artwork.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OrbSize {
    /// Chat-avatar scale (64 logical px).
    #[default]
    Avatar,
    /// Inline-text scale (20 logical px).
    Inline,
    /// Prominent status / card scale (96 logical px).
    Large,
    /// Hero / empty-state scale (128 logical px).
    Hero,
}

impl OrbSize {
    /// All sizes in compact-to-prominent gallery order.
    pub const ALL_SIZES: &'static [Self] =
        &[Self::Inline, Self::Avatar, Self::Large, Self::Hero];

    /// Logical pixel edge length of the orb.
    #[must_use]
    pub const fn pixels(self) -> f32 {
        match self {
            Self::Inline => 20.0,
            Self::Avatar => 64.0,
            Self::Large => 96.0,
            Self::Hero => 128.0,
        }
    }

    /// Stable lowercase name for controls and command-line arguments.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Avatar => "avatar",
            Self::Large => "large",
            Self::Hero => "hero",
        }
    }

    /// Compact playground label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Inline => "20 · inline",
            Self::Avatar => "64 · avatar",
            Self::Large => "96 · large",
            Self::Hero => "128 · hero",
        }
    }
}

/// Theme mode for monochrome ink on transparent canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum OrbTheme {
    /// Resolve from the window appearance (GPUI `WindowAppearance`).
    #[default]
    Auto,
    /// Light ink for dark backgrounds.
    Dark,
    /// Dark ink for light backgrounds.
    Light,
}

/// Internal mode keys - one painter per key. Grows in lockstep with
/// [`OrbState`], hence `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModeKey {
    Orbits,
    Globe,
    Rubik,
    Wave,
    Web,
    Braid,
    Ribbon,
    Ring,
    Morph,
    Focus,
    Gyroscope,
    Echo,
}

impl ModeKey {
    #[must_use]
    pub const fn from_state(state: OrbState) -> Self {
        match state {
            OrbState::Working => Self::Orbits,
            OrbState::Searching => Self::Globe,
            OrbState::Solving => Self::Rubik,
            OrbState::Listening => Self::Wave,
            OrbState::Connecting => Self::Web,
            OrbState::Weaving => Self::Braid,
            OrbState::Composing => Self::Ribbon,
            OrbState::Breathing => Self::Ring,
            OrbState::Shaping => Self::Morph,
            OrbState::Focusing => Self::Focus,
            OrbState::Reasoning => Self::Gyroscope,
            OrbState::Recalling => Self::Echo,
        }
    }
}
