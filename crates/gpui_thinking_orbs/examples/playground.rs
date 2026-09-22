//! Interactive playground for all twelve thinking-orb states.
//!
//! Run: `cargo run --example playground`

use gpui::{
    App, Bounds, Context, Entity, InteractiveElement, MouseButton,
    SharedString, Window, WindowBounds, WindowOptions, div, prelude::*, px,
    rgb, size,
};
use gpui_thinking_orbs::{OrbSize, OrbState, OrbTheme, ThinkingOrb};

struct Playground {
    orb: Entity<ThinkingOrb>,
    state: OrbState,
    size: OrbSize,
    theme: OrbTheme,
    speed: f32,
    paused: bool,
}

impl Playground {
    fn new(cx: &mut Context<'_, Self>) -> Self {
        let state = OrbState::Working;
        let size = OrbSize::Hero;
        let theme = OrbTheme::Dark;
        let speed = 1.0;
        let paused = false;
        let orb = cx.new(|_| {
            ThinkingOrb::new()
                .state(state)
                .size(size)
                .theme(theme)
                .speed(speed)
                .paused(paused)
        });
        Self {
            orb,
            state,
            size,
            theme,
            speed,
            paused,
        }
    }

    fn sync_orb(&mut self, cx: &mut Context<'_, Self>) {
        let state = self.state;
        let size = self.size;
        let theme = self.theme;
        let speed = self.speed;
        let paused = self.paused;
        self.orb.update(cx, |orb, cx| {
            orb.set_state(state, cx);
            orb.set_size(size, cx);
            orb.set_theme(theme, cx);
            orb.set_speed(speed, cx);
            orb.set_paused(paused, cx);
        });
    }

    fn cycle_state(&mut self, delta: i32, cx: &mut Context<'_, Self>) {
        let all = OrbState::ALL_STATES;
        let idx = all.iter().position(|s| *s == self.state).unwrap_or(0) as i32;
        let next = (idx + delta).rem_euclid(all.len() as i32) as usize;
        self.state = all[next];
        self.sync_orb(cx);
        cx.notify();
    }
}

impl Render for Playground {
    fn render(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        let bg = match self.theme {
            OrbTheme::Light => rgb(0xf4f4f5),
            OrbTheme::Dark | OrbTheme::Auto => rgb(0x09090b),
        };
        let fg = match self.theme {
            OrbTheme::Light => rgb(0x18181b),
            OrbTheme::Dark | OrbTheme::Auto => rgb(0xfafafa),
        };
        let muted = match self.theme {
            OrbTheme::Light => rgb(0x71717a),
            OrbTheme::Dark | OrbTheme::Auto => rgb(0xa1a1aa),
        };
        let chip_bg = match self.theme {
            OrbTheme::Light => rgb(0xe4e4e7),
            OrbTheme::Dark | OrbTheme::Auto => rgb(0x27272a),
        };
        let chip_active = rgb(0x3b82f6);
        let border = match self.theme {
            OrbTheme::Light => rgb(0xd4d4d8),
            OrbTheme::Dark | OrbTheme::Auto => rgb(0x27272a),
        };

        let state = self.state;
        let size_v = self.size;
        let theme = self.theme;
        let speed = self.speed;
        let paused = self.paused;

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(bg)
            .text_color(fg)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_6()
                    .p_8()
                    .size_full()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_2xl()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Thinking orbs"),
                            )
                            .child(
                                div().text_color(muted).text_sm().child(
                                    "GPUI port of orbs.jakubantalik.com - dotted agent status",
                                ),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .items_center()
                            .justify_center()
                            .gap_4()
                            .py_10()
                            .rounded_xl()
                            .border_1()
                            .border_color(border)
                            .child(self.orb.clone())
                            .child(div().text_color(muted).text_sm().child(format!(
                                "{}  ·  {}",
                                state.label(),
                                state.as_str()
                            ))),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(div().text_xs().text_color(muted).child("SIZE"))
                            .child(div().flex().flex_wrap().gap_2().children(
                                OrbSize::ALL_SIZES.iter().copied().map(|orb_size| {
                                    let active = orb_size == size_v;
                                    div()
                                        .id(SharedString::from(format!(
                                            "size-{}",
                                            orb_size.as_str()
                                        )))
                                        .px_3()
                                        .py_1()
                                        .rounded_md()
                                        .text_sm()
                                        .cursor_pointer()
                                        .bg(if active { chip_active } else { chip_bg })
                                        .text_color(if active { rgb(0xffffff) } else { muted })
                                        .child(orb_size.label())
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(move |this, _, _, cx| {
                                                this.size = orb_size;
                                                this.sync_orb(cx);
                                                cx.notify();
                                            }),
                                        )
                                }),
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(div().text_xs().text_color(muted).child("STATE"))
                            .child(div().flex().flex_wrap().gap_2().children(
                                OrbState::ALL_STATES.iter().copied().map(|s| {
                                    let active = s == state;
                                    let label = s.as_str();
                                    div()
                                        .id(SharedString::from(format!("state-{label}")))
                                        .px_3()
                                        .py_1()
                                        .rounded_md()
                                        .text_sm()
                                        .cursor_pointer()
                                        .bg(if active { chip_active } else { chip_bg })
                                        .text_color(if active { rgb(0xffffff) } else { muted })
                                        .child(label)
                                        .on_mouse_down(
                                            MouseButton::Left,
                                            cx.listener(move |this, _, _, cx| {
                                                this.state = s;
                                                this.sync_orb(cx);
                                                cx.notify();
                                            }),
                                        )
                                }),
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(div().text_xs().text_color(muted).child("THEME"))
                            .child(
                                div().flex().gap_2().children(
                                    [
                                        (OrbTheme::Dark, "dark"),
                                        (OrbTheme::Light, "light"),
                                        (OrbTheme::Auto, "auto"),
                                    ]
                                    .into_iter()
                                    .map(|(th, label)| {
                                        let active = th == theme;
                                        div()
                                            .id(SharedString::from(format!("theme-{label}")))
                                            .px_3()
                                            .py_1()
                                            .rounded_md()
                                            .text_sm()
                                            .cursor_pointer()
                                            .bg(if active { chip_active } else { chip_bg })
                                            .text_color(if active { rgb(0xffffff) } else { muted })
                                            .child(label)
                                            .on_mouse_down(
                                                MouseButton::Left,
                                                cx.listener(move |this, _, _, cx| {
                                                    this.theme = th;
                                                    this.sync_orb(cx);
                                                    cx.notify();
                                                }),
                                            )
                                    }),
                                ),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_4()
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(muted)
                                            .child(format!("SPEED  {speed:.2}×")),
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .id("speed-down")
                                                    .w(px(36.))
                                                    .h(px(32.))
                                                    .flex()
                                                    .items_center()
                                                    .justify_center()
                                                    .rounded_md()
                                                    .cursor_pointer()
                                                    .bg(chip_bg)
                                                    .text_color(muted)
                                                    .child("−")
                                                    .on_mouse_down(
                                                        MouseButton::Left,
                                                        cx.listener(|this, _, _, cx| {
                                                            this.speed =
                                                                (this.speed - 0.25).max(0.25);
                                                            this.sync_orb(cx);
                                                            cx.notify();
                                                        }),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .id("speed-up")
                                                    .w(px(36.))
                                                    .h(px(32.))
                                                    .flex()
                                                    .items_center()
                                                    .justify_center()
                                                    .rounded_md()
                                                    .cursor_pointer()
                                                    .bg(chip_bg)
                                                    .text_color(muted)
                                                    .child("+")
                                                    .on_mouse_down(
                                                        MouseButton::Left,
                                                        cx.listener(|this, _, _, cx| {
                                                            this.speed =
                                                                (this.speed + 0.25).min(4.0);
                                                            this.sync_orb(cx);
                                                            cx.notify();
                                                        }),
                                                    ),
                                            ),
                                    ),
                            )
                            .child(
                                div()
                                    .id("pause-btn")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .text_sm()
                                    .cursor_pointer()
                                    .bg(if paused { chip_active } else { chip_bg })
                                    .text_color(if paused { rgb(0xffffff) } else { muted })
                                    .child(if paused { "resume" } else { "pause" })
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.paused = !this.paused;
                                            this.sync_orb(cx);
                                            cx.notify();
                                        }),
                                    ),
                            )
                            .child(
                                div()
                                    .id("next-state")
                                    .px_3()
                                    .py_1()
                                    .rounded_md()
                                    .text_sm()
                                    .cursor_pointer()
                                    .bg(chip_bg)
                                    .text_color(muted)
                                    .child("next state →")
                                    .on_mouse_down(
                                        MouseButton::Left,
                                        cx.listener(|this, _, _, cx| {
                                            this.cycle_state(1, cx);
                                        }),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(muted)
                            .child("MIT · geometry ported from Jakub Antalik's thinking-orbs"),
                    ),
            )
    }
}

fn main() {
    gpui_kit::application().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(560.), px(720.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(gpui::TitlebarOptions {
                    title: Some("gpui-thinking-orbs".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            |_, cx| cx.new(Playground::new),
        )
        .unwrap();
        cx.activate(true);
    });
}
