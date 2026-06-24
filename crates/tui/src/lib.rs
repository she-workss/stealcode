use std::{
    io::Write as _,
    time::{Duration, Instant},
};

use anyhow::Result;
use ratatui::{
    Frame, Terminal,
    backend::TerminaBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    termina::{
        EventReader, PlatformTerminal, Terminal as _,
        escape::{
            csi::{self, Csi},
            osc::{ColorOrQuery, DynamicColorNumber, Osc},
        },
        event::{Event, KeyCode, KeyEventKind, Modifiers},
        style::RgbColor,
    },
    widgets::{Block, Paragraph},
};
use settings::Settings;

type AppTerminal = Terminal<TerminaBackend<PlatformTerminal>>;

const COLOR_QUERY_TIMEOUT: Duration = Duration::from_millis(250);

macro_rules! decset {
    ($mode:ident) => {{
        let mode = csi::DecPrivateMode::Code(csi::DecPrivateModeCode::$mode);
        Csi::Mode(csi::Mode::SetDecPrivateMode(mode))
    }};
}

macro_rules! decreset {
    ($mode:ident) => {{
        let mode = csi::DecPrivateMode::Code(csi::DecPrivateModeCode::$mode);
        Csi::Mode(csi::Mode::ResetDecPrivateMode(mode))
    }};
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    Quit,
    OpenPalette,
    Search,
    NextTheme,
    RenameTitle,
    ShowHelp,
    Unknown(String),
}

fn dispatch(event: &Event) -> Action {
    let Event::Key(key) = event else {
        return Action::Unknown(format!("{event:?}"));
    };
    if key.kind == KeyEventKind::Release {
        return Action::Unknown(String::new());
    }
    if key.modifiers == Modifiers::CONTROL {
        match key.code {
            KeyCode::Char('c') => return Action::Quit,
            KeyCode::Char('p') => return Action::OpenPalette,
            KeyCode::Char('f') => return Action::Search,
            KeyCode::Char('t') => return Action::RenameTitle,
            KeyCode::Char('h') => return Action::ShowHelp,
            _ => {}
        }
    }
    if key.code == KeyCode::Tab {
        return Action::NextTheme;
    }
    Action::Unknown(format!("{key:?}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OriginalColors {
    fg: Option<RgbColor>,
    bg: Option<RgbColor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorMode {
    Dynamic(OriginalColors),
    SgrOnly,
}

impl ColorMode {
    const fn supports_osc(self) -> bool {
        matches!(self, Self::Dynamic(_))
    }
}

/// Converts termina's [`RgbColor`] into the [`Color`] ratatui widgets style
/// with.
///
/// `RgbColor` already carries everything a color needs (`red`/`green`/`blue` as
/// plain `u8`s), and it's the exact type the `OSC 10`/`OSC 11` calls in this
/// module take. There's no reason to keep a second, ratatui-flavored copy of
/// the same three bytes sitting next to it — that's just two places that can
/// drift out of sync. This is the only conversion point, called wherever a
/// `Style` needs a `Color`; everywhere else just keeps passing the `RgbColor`
/// straight through.
///
/// Implemented as a local extension trait rather than `From`/`Into`, since
/// neither `RgbColor` nor `Color` is defined in this crate and the orphan rules
/// don't allow implementing a foreign trait for a foreign type.
trait RgbColorExt {
    fn to_color(self) -> Color;
}

impl RgbColorExt for RgbColor {
    fn to_color(self) -> Color {
        Color::Rgb(self.red, self.green, self.blue)
    }
}

fn extract_queried_color(
    event: &Event,
    expected: DynamicColorNumber,
) -> Option<RgbColor> {
    let Event::Osc(Osc::ChangeDynamicColors(number, values)) = event else {
        return None;
    };
    if *number != expected {
        return None;
    }
    values.iter().find_map(|v| match v {
        ColorOrQuery::Color(rgb) => Some(*rgb),
        ColorOrQuery::Query => None,
    })
}

fn detect_color_mode(
    terminal: &mut AppTerminal,
    events: &EventReader,
) -> Result<ColorMode> {
    let backend = terminal.backend_mut();
    write!(
        backend,
        "{}{}",
        Osc::ChangeDynamicColors(
            DynamicColorNumber::TextForegroundColor,
            vec![ColorOrQuery::Query],
        ),
        Osc::ChangeDynamicColors(
            DynamicColorNumber::TextBackgroundColor,
            vec![ColorOrQuery::Query],
        ),
    )?;
    backend.flush()?;
    let mut orig = OriginalColors { fg: None, bg: None };
    let deadline = Instant::now() + COLOR_QUERY_TIMEOUT;
    while orig.fg.is_none() || orig.bg.is_none() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let filter = |event: &Event| {
            matches!(
                event,
                Event::Osc(Osc::ChangeDynamicColors(
                    DynamicColorNumber::TextForegroundColor
                        | DynamicColorNumber::TextBackgroundColor,
                    _,
                ))
            )
        };
        if !events.poll(Some(remaining), filter)? {
            break;
        }
        let event = events.read(filter)?;
        if let Some(rgb) = extract_queried_color(
            &event,
            DynamicColorNumber::TextForegroundColor,
        ) {
            orig.fg = Some(rgb);
        } else if let Some(rgb) = extract_queried_color(
            &event,
            DynamicColorNumber::TextBackgroundColor,
        ) {
            orig.bg = Some(rgb);
        }
    }

    if orig.fg.is_some() || orig.bg.is_some() {
        Ok(ColorMode::Dynamic(orig))
    } else {
        Ok(ColorMode::SgrOnly)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ColorScheme {
    name: &'static str,
    fg: RgbColor,
    bg: RgbColor,
    accent: RgbColor,
    border: RgbColor,
    title_bg: RgbColor,
}

const SCHEME_DEFAULT: ColorScheme = ColorScheme {
    name: "Default",
    fg: RgbColor::new(204, 204, 204),
    bg: RgbColor::new(0, 0, 0),
    accent: RgbColor::new(0, 200, 80),
    border: RgbColor::new(80, 80, 80),
    title_bg: RgbColor::new(0, 120, 50),
};

const SCHEME_OCEAN: ColorScheme = ColorScheme {
    name: "Ocean",
    fg: RgbColor::new(180, 210, 255),
    bg: RgbColor::new(10, 20, 40),
    accent: RgbColor::new(100, 200, 255),
    border: RgbColor::new(40, 70, 120),
    title_bg: RgbColor::new(30, 80, 160),
};

const SCHEME_FOREST: ColorScheme = ColorScheme {
    name: "Forest",
    fg: RgbColor::new(200, 230, 180),
    bg: RgbColor::new(10, 30, 10),
    accent: RgbColor::new(120, 220, 80),
    border: RgbColor::new(40, 80, 30),
    title_bg: RgbColor::new(50, 100, 40),
};

const SCHEME_SUNSET: ColorScheme = ColorScheme {
    name: "Sunset",
    fg: RgbColor::new(255, 220, 180),
    bg: RgbColor::new(40, 10, 30),
    accent: RgbColor::new(255, 140, 60),
    border: RgbColor::new(120, 40, 60),
    title_bg: RgbColor::new(160, 50, 70),
};

const SCHEMES: &[ColorScheme] =
    &[SCHEME_DEFAULT, SCHEME_OCEAN, SCHEME_FOREST, SCHEME_SUNSET];

impl ColorScheme {
    fn block_style(self) -> Style {
        Style::default()
            .fg(self.fg.to_color())
            .bg(self.bg.to_color())
    }

    fn accent_style(self) -> Style {
        Style::default()
            .fg(self.accent.to_color())
            .bg(self.bg.to_color())
            .add_modifier(Modifier::BOLD)
    }

    fn title_style(self) -> Style {
        Style::default()
            .fg(self.fg.to_color())
            .bg(self.title_bg.to_color())
            .add_modifier(Modifier::BOLD)
    }

    fn border_style(self) -> Style {
        Style::default().fg(self.border.to_color())
    }

    fn hint_style(self) -> Style {
        Style::default().fg(Color::DarkGray).bg(self.bg.to_color())
    }

    fn bg_style(self) -> Style {
        Style::default().bg(self.bg.to_color())
    }

    fn apply_osc(self, terminal: &mut AppTerminal) -> Result<()> {
        let backend = terminal.backend_mut();
        write!(
            backend,
            "{}{}",
            Osc::ChangeDynamicColors(
                DynamicColorNumber::TextForegroundColor,
                vec![ColorOrQuery::Color(self.fg)],
            ),
            Osc::ChangeDynamicColors(
                DynamicColorNumber::TextBackgroundColor,
                vec![ColorOrQuery::Color(self.bg)],
            ),
        )?;
        backend.flush()?;
        Ok(())
    }
}

struct AppState {
    event_raw: String,
    action_label: String,
    title: String,
    scheme_idx: usize,
    color_mode: ColorMode,
    mouse_pos: Option<(u16, u16)>,
    counter: u32,
}

impl AppState {
    fn new(color_mode: ColorMode) -> Self {
        Self {
            event_raw: String::from("Press any key"),
            action_label: String::from("-"),
            title: String::from("ratatui + termina"),
            scheme_idx: 0,
            color_mode,
            mouse_pos: None,
            counter: 0,
        }
    }

    fn scheme(&self) -> &ColorScheme {
        &SCHEMES[self.scheme_idx]
    }

    fn next_counter(&mut self) -> u32 {
        let n = self.counter;
        self.counter += 1;
        n
    }

    const fn mode_label(&self) -> &'static str {
        match self.color_mode {
            ColorMode::Dynamic(_) => "OSC",
            ColorMode::SgrOnly => "SGR",
        }
    }
}

fn handle_action(
    action: &Action,
    state: &mut AppState,
    terminal: &mut AppTerminal,
) -> Result<bool> {
    match action {
        Action::Quit => return Ok(true),
        Action::OpenPalette => {
            state.action_label = "OpenCommandPalette".into();
        }
        Action::Search => {
            state.action_label = "Search".into();
        }
        Action::NextTheme => {
            state.scheme_idx = (state.scheme_idx + 1) % SCHEMES.len();
            if state.color_mode.supports_osc() {
                state.scheme().apply_osc(terminal)?;
            }
            state.action_label = format!(
                "Theme: {} ({})",
                state.scheme().name,
                state.mode_label(),
            );
        }
        Action::RenameTitle => {
            let n = state.next_counter();
            state.title = format!("ratatui + termina [#{n}]");
            let backend = terminal.backend_mut();
            write!(
                backend,
                "{}",
                Osc::SetIconNameAndWindowTitle(&state.title),
            )?;
            backend.flush()?;
            state.action_label = format!("Title: \"{}\"", state.title);
        }
        Action::ShowHelp => {
            state.action_label = "Ctrl+P/F/H/T/?/C".into();
        }
        Action::Unknown(raw) => {
            if !raw.is_empty() {
                state.action_label = format!("Unknown: {raw}");
            }
        }
    }
    Ok(false)
}

fn init_terminal() -> Result<(AppTerminal, EventReader)> {
    let mut output = PlatformTerminal::new()?;
    output.enter_raw_mode()?;
    write!(
        output,
        "{}{}{}{}{}{}{}{}",
        decset!(ClearAndEnableAlternateScreen),
        decreset!(ShowCursor),
        Csi::Window(Box::new(csi::Window::PushIconAndWindowTitle)),
        Osc::SetIconNameAndWindowTitle("ratatui + termina"),
        decset!(MouseTracking),
        decset!(ButtonEventMouse),
        decset!(AnyEventMouse),
        decset!(SGRMouse),
    )?;
    output.flush()?;
    let events = output.event_reader();
    let backend = TerminaBackend::new(output);
    let terminal = Terminal::new(backend)?;
    Ok((terminal, events))
}

fn restore_terminal(
    terminal: &mut AppTerminal,
    color_mode: ColorMode,
) -> Result<()> {
    let backend = terminal.backend_mut();
    let fg_osc;
    let bg_osc;
    match color_mode {
        ColorMode::Dynamic(orig) => {
            fg_osc = match orig.fg {
                Some(rgb) => Osc::ChangeDynamicColors(
                    DynamicColorNumber::TextForegroundColor,
                    vec![ColorOrQuery::Color(rgb)],
                ),
                None => Osc::ResetDynamicColor(
                    DynamicColorNumber::TextForegroundColor,
                ),
            };
            bg_osc = match orig.bg {
                Some(rgb) => Osc::ChangeDynamicColors(
                    DynamicColorNumber::TextBackgroundColor,
                    vec![ColorOrQuery::Color(rgb)],
                ),
                None => Osc::ResetDynamicColor(
                    DynamicColorNumber::TextBackgroundColor,
                ),
            };
        }
        ColorMode::SgrOnly => {
            fg_osc =
                Osc::ResetDynamicColor(DynamicColorNumber::TextForegroundColor);
            bg_osc =
                Osc::ResetDynamicColor(DynamicColorNumber::TextBackgroundColor);
        }
    }
    write!(
        backend,
        "{}{}{}{}{}{}{}{}",
        decset!(ShowCursor),
        decreset!(ClearAndEnableAlternateScreen),
        Csi::Window(Box::new(csi::Window::PopIconAndWindowTitle)),
        fg_osc,
        bg_osc,
        decreset!(MouseTracking),
        decreset!(ButtonEventMouse),
        decreset!(AnyEventMouse),
    )?;
    backend.flush()?;
    Ok(())
}

pub fn run_tui(_settings: &Settings) -> Result<()> {
    let (mut terminal, events) = init_terminal()?;
    let color_mode = detect_color_mode(&mut terminal, &events)?;
    let mut state = AppState::new(color_mode);
    if color_mode.supports_osc() {
        state.scheme().apply_osc(&mut terminal)?;
    }
    let result = run(&mut terminal, &events, &mut state);
    restore_terminal(&mut terminal, color_mode)?;
    result
}

fn run(
    terminal: &mut AppTerminal,
    events: &EventReader,
    state: &mut AppState,
) -> Result<()> {
    loop {
        terminal.draw(|f| render(f, state))?;
        let event = events.read(|_| true)?;
        if let Event::Mouse(mouse) = &event {
            state.event_raw = format!("{mouse:?}");
            state.mouse_pos = Some((mouse.column, mouse.row));
            state.action_label = format!(
                "Mouse: kind={:?} col={} row={}",
                mouse.kind, mouse.column, mouse.row,
            );
            continue;
        }
        let raw = format!("{event:?}");
        let action = dispatch(&event);
        if !matches!(action, Action::Unknown(ref s) if s.is_empty()) {
            state.event_raw = raw;
        }
        if handle_action(&action, state, terminal)? {
            break;
        }
    }
    Ok(())
}

fn render(f: &mut Frame<'_>, state: &AppState) {
    let s = *state.scheme();
    f.render_widget(Block::default().style(s.bg_style()), f.area());
    let chunks = main_layout(f.area());
    render_panel(
        f,
        chunks[0],
        "Termina Event",
        &state.event_raw,
        s.block_style(),
        s,
    );
    render_panel(
        f,
        chunks[1],
        "Action",
        &state.action_label,
        s.accent_style(),
        s,
    );
    render_panel(
        f,
        chunks[2],
        "State",
        &format!(
            "Title: \"{}\"  Theme: {}  Mode: {}",
            state.title,
            s.name,
            state.mode_label(),
        ),
        s.block_style(),
        s,
    );
    let mouse = match state.mouse_pos {
        Some((x, y)) => format!("Mouse: x={x} y={y}"),
        None => "Mouse: -".into(),
    };
    render_panel(f, chunks[3], "Mouse", &mouse, s.accent_style(), s);
    f.render_widget(
        Paragraph::new(format!(
            "Ctrl+P (Palette) Ctrl+F (Search) Ctrl+T (Title++) \
             Tab (Next Theme) Ctrl+H (Help) Ctrl+C (Exit) | {}",
            state.mode_label(),
        ))
        .style(s.hint_style()),
        chunks[4],
    );
}

fn main_layout(area: Rect) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area)
        .to_vec()
}

fn render_panel(
    f: &mut Frame<'_>,
    area: Rect,
    title: &str,
    content: &str,
    content_style: Style,
    s: ColorScheme,
) {
    let block = Block::bordered()
        .title(title)
        .title_style(s.title_style())
        .border_style(s.border_style());
    f.render_widget(
        Paragraph::new(content).style(content_style).block(block),
        area,
    );
}

#[cfg(test)]
mod tests {
    use ratatui::termina::event::KeyEvent;

    use super::*;

    #[test]
    fn extract_color_from_matching_slot() {
        let event = Event::Osc(Osc::ChangeDynamicColors(
            DynamicColorNumber::TextForegroundColor,
            vec![ColorOrQuery::Color(RgbColor::new(10, 20, 30))],
        ));
        assert_eq!(
            extract_queried_color(
                &event,
                DynamicColorNumber::TextForegroundColor
            ),
            Some(RgbColor::new(10, 20, 30)),
        );
    }

    #[test]
    fn extract_ignores_wrong_slot() {
        let event = Event::Osc(Osc::ChangeDynamicColors(
            DynamicColorNumber::TextForegroundColor,
            vec![ColorOrQuery::Color(RgbColor::new(10, 20, 30))],
        ));
        assert_eq!(
            extract_queried_color(
                &event,
                DynamicColorNumber::TextBackgroundColor
            ),
            None,
        );
    }

    #[test]
    fn extract_ignores_unanswered_query() {
        let event = Event::Osc(Osc::ChangeDynamicColors(
            DynamicColorNumber::TextForegroundColor,
            vec![ColorOrQuery::Query],
        ));
        assert_eq!(
            extract_queried_color(
                &event,
                DynamicColorNumber::TextForegroundColor
            ),
            None,
        );
    }

    #[test]
    fn sgr_only_mode_has_no_osc_support() {
        assert!(!ColorMode::SgrOnly.supports_osc());
    }

    #[test]
    fn dynamic_mode_has_osc_support() {
        let mode = ColorMode::Dynamic(OriginalColors { fg: None, bg: None });
        assert!(mode.supports_osc());
    }

    #[test]
    fn dispatch_quit() {
        let event =
            Event::Key(KeyEvent::new(KeyCode::Char('c'), Modifiers::CONTROL));
        assert_eq!(dispatch(&event), Action::Quit);
    }

    #[test]
    fn dispatch_palette() {
        let event =
            Event::Key(KeyEvent::new(KeyCode::Char('p'), Modifiers::CONTROL));
        assert_eq!(dispatch(&event), Action::OpenPalette);
    }

    #[test]
    fn dispatch_help() {
        let event =
            Event::Key(KeyEvent::new(KeyCode::Char('h'), Modifiers::CONTROL));
        assert_eq!(dispatch(&event), Action::ShowHelp);
    }

    #[test]
    fn dispatch_next_theme_via_tab() {
        let event = Event::Key(KeyEvent::new(KeyCode::Tab, Modifiers::NONE));
        assert_eq!(dispatch(&event), Action::NextTheme);
    }

    #[test]
    fn dispatch_plain_char_is_unknown() {
        let event =
            Event::Key(KeyEvent::new(KeyCode::Char('x'), Modifiers::NONE));
        assert_eq!(
            dispatch(&event),
            Action::Unknown(format!(
                "{:?}",
                KeyEvent::new(KeyCode::Char('x'), Modifiers::NONE)
            ))
        );
    }

    #[test]
    fn rgb_color_converts_to_matching_ratatui_color() {
        let rgb = RgbColor::new(180, 210, 255);
        assert_eq!(rgb.to_color(), Color::Rgb(180, 210, 255));
    }

    #[test]
    fn color_scheme_fg_and_bg_drive_block_style() {
        let scheme = SCHEME_OCEAN;
        let style = scheme.block_style();
        assert_eq!(style.fg, Some(scheme.fg.to_color()));
        assert_eq!(style.bg, Some(scheme.bg.to_color()));
    }

    #[test]
    fn scheme_cycle() {
        assert_eq!(SCHEMES[0].name, "Default");
        assert_eq!(SCHEMES[3].name, "Sunset");
        let idx = (3 + 1) % SCHEMES.len();
        assert_eq!(SCHEMES[idx].name, "Default");
    }
}
