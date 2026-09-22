//! GPUI `ThinkingOrb` view - continuous animated status indicator.

use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

use gpui::{
    App, AppContext, Bounds, Context, IntoElement, ParentElement, Pixels,
    Render, Styled, Task, Window, WindowAppearance, canvas, div, px,
};

use crate::{
    engine::{Frame, draw_mode_into_resolved},
    paint::paint_frame,
    presets::{Resolved, resolve_preset},
    types::{OrbSize, OrbState, OrbTheme},
};

/// Frames per second the orb redraws at when no explicit rate is set.
///
/// The orb is a status indicator, not a game: its motion is slow and organic,
/// and at 30 fps it is indistinguishable from 60 while costing half as much.
/// Every redraw walks the whole element tree, so the tick rate - not the
/// geometry - is what dominates CPU.
pub const DEFAULT_TARGET_FPS: f32 = 30.0;

/// Animated thinking-orb status indicator for AI / agent UIs.
///
/// ```ignore
/// cx.new(|_cx| ThinkingOrb::new().state(OrbState::Searching))
/// ```
#[derive(Debug)]
pub struct ThinkingOrb {
    state: OrbState,
    size: OrbSize,
    theme: OrbTheme,
    /// Multiplier on top of the preset's baked speed.
    speed: f32,
    paused: bool,
    /// When true, freeze on a static representative frame (`t = 0.6`).
    reduced_motion: bool,
    /// Redraw rate ceiling.
    target_fps: f32,
    /// Stop animating while the host window is not the active one.
    pause_when_inactive: bool,
    /// Host-controlled visibility. When false the timer is cancelled and the
    /// orb freezes - use this when the entity is still mounted but scrolled
    /// off-screen (GPUI has no intersection observer).
    visible: bool,

    // ---- clock ----
    started: Instant,
    /// Wall time accumulated while paused, subtracted from the animation clock
    /// so pausing genuinely freezes motion instead of merely stopping redraws.
    paused_total: Duration,
    /// Set while a pause is in effect.
    paused_at: Option<Instant>,

    // ---- caches ----
    /// `(state, size)` the cached `resolved` was computed for.
    cache_key: (OrbState, OrbSize),
    resolved: Resolved,
    /// Geometry buffer reused across frames. Behind `Rc<RefCell<_>>` because
    /// the canvas paint callback must be `'static` and so cannot borrow
    /// `self`.
    frame: Rc<RefCell<Frame>>,
    /// Explicit invalidation for animation ticks and semantic changes. Parent
    /// renders between ticks reuse the retained geometry.
    geometry_dirty: bool,

    /// Pending redraw timer. At most one stays in flight; dropping it cancels
    /// animation immediately when the orb becomes paused or invisible.
    tick: Option<Task<()>>,
    /// Window-activation subscription, registered lazily on first render since
    /// it needs a `Window`. Dropping it unsubscribes.
    activation: Option<gpui::Subscription>,
    /// OS appearance subscription - only while [`OrbTheme::Auto`] is selected.
    appearance: Option<gpui::Subscription>,
}

impl Default for ThinkingOrb {
    fn default() -> Self {
        Self::new()
    }
}

impl ThinkingOrb {
    pub fn new() -> Self {
        let state = OrbState::Working;
        let size = OrbSize::Avatar;
        Self {
            state,
            size,
            theme: OrbTheme::Auto,
            speed: 1.0,
            paused: false,
            reduced_motion: false,
            target_fps: DEFAULT_TARGET_FPS,
            pause_when_inactive: true,
            visible: true,
            started: Instant::now(),
            paused_total: Duration::ZERO,
            paused_at: None,
            cache_key: (state, size),
            resolved: resolve_preset(state, size),
            frame: Rc::new(RefCell::new(Frame::new())),
            geometry_dirty: true,
            tick: None,
            activation: None,
            appearance: None,
        }
    }

    pub fn state(mut self, state: OrbState) -> Self {
        self.state = state;
        self
    }

    pub fn size(mut self, size: OrbSize) -> Self {
        self.size = size;
        self
    }

    pub fn theme(mut self, theme: OrbTheme) -> Self {
        self.theme = theme;
        self
    }

    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = sanitize_speed(speed);
        self
    }

    /// Same clock rules as [`Self::set_paused`]: entering pause records
    /// `paused_at`; leaving folds the elapsed pause into `paused_total`.
    pub fn paused(mut self, paused: bool) -> Self {
        apply_pause_clock(
            &mut self.paused,
            &mut self.paused_at,
            &mut self.paused_total,
            paused,
        );
        self
    }

    pub fn reduced_motion(mut self, reduced: bool) -> Self {
        self.reduced_motion = reduced;
        self
    }

    /// Cap the redraw rate. Values are clamped to `1.0..=240.0`.
    ///
    /// Lower is cheaper: cost scales linearly with this number.
    pub fn target_fps(mut self, fps: f32) -> Self {
        self.target_fps = sanitize_fps(fps);
        self
    }

    /// Whether to freeze while the host window is inactive. Defaults to `true`.
    ///
    /// A background window's animation is not visible to anyone, so this is
    /// usually free. Set it to `false` if the orb must keep moving in a window
    /// that is visible but unfocused - a side panel, or a floating HUD.
    pub fn pause_when_inactive(mut self, pause: bool) -> Self {
        self.pause_when_inactive = pause;
        self
    }

    /// Whether the host considers this orb on-screen. Defaults to `true`.
    ///
    /// GPUI 0.2 has no intersection observer. When you keep the entity mounted
    /// in a scrollable list but it has scrolled away, call
    /// [`Self::set_visible`]`(false)` (or build with `.visible(false)`) so the
    /// timer stops. Prefer unmounting when you can.
    pub fn visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Mutable setters for interactive playgrounds.
    pub fn set_state(&mut self, state: OrbState, cx: &mut Context<'_, Self>) {
        if self.state != state {
            self.state = state;
            self.geometry_dirty = true;
            cx.notify();
        }
    }

    pub fn set_size(&mut self, size: OrbSize, cx: &mut Context<'_, Self>) {
        if self.size != size {
            self.size = size;
            self.geometry_dirty = true;
            cx.notify();
        }
    }

    pub fn set_theme(&mut self, theme: OrbTheme, cx: &mut Context<'_, Self>) {
        if self.theme != theme {
            self.theme = theme;
            // Drop the appearance subscription when leaving Auto; it is
            // re-registered lazily on the next render if needed.
            if theme != OrbTheme::Auto {
                self.appearance = None;
            }
            cx.notify();
        }
    }

    pub fn set_speed(&mut self, speed: f32, cx: &mut Context<'_, Self>) {
        let speed = sanitize_speed(speed);
        if self.speed != speed {
            self.speed = speed;
            self.geometry_dirty = true;
            cx.notify();
        }
    }

    pub fn set_paused(&mut self, paused: bool, cx: &mut Context<'_, Self>) {
        if self.paused == paused {
            return;
        }
        apply_pause_clock(
            &mut self.paused,
            &mut self.paused_at,
            &mut self.paused_total,
            paused,
        );
        self.geometry_dirty = true;
        cx.notify();
    }

    pub fn set_reduced_motion(
        &mut self,
        reduced: bool,
        cx: &mut Context<'_, Self>,
    ) {
        if self.reduced_motion != reduced {
            self.reduced_motion = reduced;
            self.geometry_dirty = true;
            cx.notify();
        }
    }

    pub fn set_target_fps(&mut self, fps: f32, cx: &mut Context<'_, Self>) {
        self.target_fps = sanitize_fps(fps);
        cx.notify();
    }

    pub fn set_pause_when_inactive(
        &mut self,
        pause: bool,
        cx: &mut Context<'_, Self>,
    ) {
        if self.pause_when_inactive != pause {
            self.pause_when_inactive = pause;
            cx.notify();
        }
    }

    /// Host visibility gate - see [`Self::visible`].
    pub fn set_visible(&mut self, visible: bool, cx: &mut Context<'_, Self>) {
        if self.visible != visible {
            self.visible = visible;
            self.geometry_dirty = true;
            cx.notify();
        }
    }

    pub fn pause_when_inactive_value(&self) -> bool {
        self.pause_when_inactive
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn state_value(&self) -> OrbState {
        self.state
    }

    pub fn size_value(&self) -> OrbSize {
        self.size
    }

    pub fn theme_value(&self) -> OrbTheme {
        self.theme
    }

    pub fn speed_value(&self) -> f32 {
        self.speed
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn reduced_motion_value(&self) -> bool {
        self.reduced_motion
    }

    pub fn target_fps_value(&self) -> f32 {
        self.target_fps
    }

    fn resolve_dark(&self, window: &Window) -> bool {
        match self.theme {
            OrbTheme::Dark => true,
            OrbTheme::Light => false,
            OrbTheme::Auto => matches!(
                window.appearance(),
                WindowAppearance::Dark | WindowAppearance::VibrantDark
            ),
        }
    }

    /// Animation clock in seconds, excluding any time spent paused.
    ///
    /// Accumulated in `f64` and only narrowed at the end: `f32` has a 24-bit
    /// mantissa, so a clock driven straight off wall time quantises visibly
    /// after several hours of uptime. Excluding paused time also means an orb
    /// that is idle most of the session barely advances its clock at all.
    ///
    /// Known limit: the engine takes `t: f32`, so very long *continuous*
    /// animation still loses step resolution. There is no seamless wrap point -
    /// the modes mix incommensurate frequencies, so folding the clock would
    /// trade slow degradation for a visible jump.
    fn time_seconds(&self) -> f32 {
        if self.reduced_motion {
            return 0.6;
        }
        let paused = match self.paused_at {
            Some(at) => self.paused_total + at.elapsed(),
            None => self.paused_total,
        };
        let live = self.started.elapsed().saturating_sub(paused);
        (live.as_secs_f64() * self.resolved.speed as f64 * self.speed as f64)
            as f32
    }

    /// Whether the orb should keep redrawing itself.
    fn animating(&self, window: &Window) -> bool {
        if !self.visible || self.paused || self.reduced_motion {
            return false;
        }
        !self.pause_when_inactive || window.is_window_active()
    }

    /// Queue the next redraw, honouring [`Self::target_fps`].
    ///
    /// This replaces `window.request_animation_frame()`, which re-notifies at
    /// the display's refresh rate - 60 Hz or more - with no way to opt down.
    fn schedule_tick(&mut self, cx: &mut Context<'_, Self>) {
        // Parent renders may happen between animation frames. Keep the timer
        // already in flight rather than cancel/reallocate it and push the next
        // frame farther into the future.
        if self.tick.is_some() {
            return;
        }
        let period = Duration::from_secs_f32(1.0 / self.target_fps);
        self.tick = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(period).await;
            let _ = this.update(cx, |orb, cx| {
                orb.tick = None;
                orb.geometry_dirty = true;
                cx.notify();
            });
        }));
    }
}

/// Shared pause-clock rules for the builder and the mutable setter.
fn apply_pause_clock(
    paused: &mut bool,
    paused_at: &mut Option<Instant>,
    paused_total: &mut Duration,
    want: bool,
) {
    if *paused == want {
        // Still clear a stuck paused_at if someone left it set while unpaused.
        if !want {
            if let Some(at) = paused_at.take() {
                *paused_total += at.elapsed();
            }
        }
        return;
    }
    *paused = want;
    if want {
        if paused_at.is_none() {
            *paused_at = Some(Instant::now());
        }
    } else if let Some(at) = paused_at.take() {
        *paused_total += at.elapsed();
    }
}

fn sanitize_speed(speed: f32) -> f32 {
    if speed.is_finite() {
        speed.clamp(0.0, 100.0)
    } else {
        1.0
    }
}

fn sanitize_fps(fps: f32) -> f32 {
    if fps.is_finite() {
        fps.clamp(1.0, 240.0)
    } else {
        DEFAULT_TARGET_FPS
    }
}

impl Render for ThinkingOrb {
    fn render(
        &mut self,
        window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        // An orb in a background window stops ticking entirely, so it needs a
        // nudge to start again when the window comes back. Registered once,
        // here, because it needs a `Window`.
        if self.activation.is_none() {
            self.activation =
                Some(cx.observe_window_activation(window, |orb, _, cx| {
                    orb.geometry_dirty = true;
                    cx.notify();
                }));
        }

        // `OrbTheme::Auto` must re-render when the OS switches light/dark, even
        // if the orb is paused or reduced-motion (no tick would fire
        // otherwise). GPUI's appearance_changed does not refresh the
        // window on its own.
        if self.theme == OrbTheme::Auto {
            if self.appearance.is_none() {
                self.appearance =
                    Some(cx.observe_window_appearance(window, |_, _, cx| {
                        cx.notify()
                    }));
            }
        } else {
            self.appearance = None;
        }

        // Presets are pure functions of (state, size); recompute only when one
        // of those actually changes rather than on every frame.
        let key = (self.state, self.size);
        if self.cache_key != key {
            self.cache_key = key;
            self.resolved = resolve_preset(self.state, self.size);
        }

        let size_px = self.size.pixels();
        let dark = self.resolve_dark(window);
        let t = self.time_seconds();
        let r_min = self.resolved.opts.r_min.unwrap_or(0.3);
        let mode = self.resolved.mode;

        // Only ticks and semantic changes invalidate geometry. A parent can
        // re-render much faster than this orb's target FPS; those extra renders
        // reuse the retained frame rather than running animation math again.
        if self.geometry_dirty {
            draw_mode_into_resolved(
                mode,
                size_px,
                t,
                &self.resolved.opts,
                &mut self.frame.borrow_mut(),
            );
            self.geometry_dirty = false;
        }

        if self.animating(window) {
            self.schedule_tick(cx);
        } else {
            // Drop any in-flight tick so a paused, hidden, or backgrounded orb
            // costs nothing at all.
            self.tick = None;
        }

        let frame = self.frame.clone();
        div()
            .size(px(size_px))
            .flex_shrink_0()
            .overflow_hidden()
            .child(
                canvas(
                    move |_bounds: Bounds<Pixels>, _window, _cx| (),
                    move |bounds, (), window, _cx| {
                        paint_frame(
                            window,
                            bounds,
                            &frame.borrow(),
                            dark,
                            r_min,
                        );
                    },
                )
                .size_full(),
            )
    }
}

/// Convenience: spawn a `ThinkingOrb` entity in the given context.
pub fn thinking_orb(cx: &mut App) -> gpui::Entity<ThinkingOrb> {
    cx.new(|_| ThinkingOrb::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_paused_false_clears_pause_clock() {
        let orb = ThinkingOrb::new().paused(true).paused(false);
        assert!(!orb.is_paused());
        assert!(orb.paused_at.is_none());
    }

    #[test]
    fn builder_paused_true_records_paused_at() {
        let orb = ThinkingOrb::new().paused(true);
        assert!(orb.is_paused());
        assert!(orb.paused_at.is_some());
    }
}
