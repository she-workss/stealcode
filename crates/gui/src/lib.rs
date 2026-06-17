use gpui::{
    SharedString, TitlebarOptions, Window, WindowOptions, div, prelude::*,
};
use gpui_component::{
    Root, StyledExt,
    button::{Button, ButtonVariants},
};
use settings::Settings;
use tracing::error;

#[derive(Debug)]
struct App;

impl Render for App {
    fn render(
        &mut self,
        _: &mut Window,
        _: &mut Context<'_, Self>,
    ) -> impl IntoElement {
        div()
            .v_flex()
            .gap_2()
            .size_full()
            .items_center()
            .justify_center()
            .child("Hello, World!")
            .child(Button::new("btn").primary().label("Click"))
    }
}

pub fn run_desktop(_settings: &Settings) -> anyhow::Result<()> {
    gpui_platform::application().run(move |cx| {
        gpui_component::init(cx);
        cx.spawn(async move |cx| {
            cx.open_window(
                WindowOptions {
                    titlebar: Some(TitlebarOptions {
                        title: Some(SharedString::from("StealCode")),
                        appears_transparent: false,
                        traffic_light_position: None,
                    }),
                    ..Default::default()
                },
                |window, cx| {
                    let view = cx.new(|_| App);
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .inspect_err(|e| error!("failed to open window: {e:?}"))
        })
        .detach();
    });
    Ok(())
}
