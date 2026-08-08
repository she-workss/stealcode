use std::{
    io::Write as _,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::Result;
use ratatui::{
    Frame, Terminal,
    backend::TerminaBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    termina::{
        EventReader, PlatformTerminal, Terminal as _,
        escape::{
            csi::{self, Csi},
            osc::{ColorOrQuery, DynamicColorNumber, Osc},
        },
        event::{
            Event, KeyCode, KeyEventKind, Modifiers, MouseButton,
            MouseEventKind,
        },
        style::RgbColor,
    },
    widgets::{Block, Paragraph},
};
use settings::Settings;
use sound::{engine, sounds::SoundName};
#[cfg(feature = "voice")]
use voice::VoiceManager;

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
    TogglePushToTalk,
    CheckForUpdates,
    ToggleAutoUpdate,
    UpdateAndRestart,
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
            KeyCode::Char('g') => return Action::TogglePushToTalk,
            KeyCode::Char('u') => return Action::CheckForUpdates,
            KeyCode::Char('a') => return Action::ToggleAutoUpdate,
            KeyCode::Char('r') => return Action::UpdateAndRestart,
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
            vec![ColorOrQuery::Query]
        ),
        Osc::ChangeDynamicColors(
            DynamicColorNumber::TextBackgroundColor,
            vec![ColorOrQuery::Query]
        )
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
                    _
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
                vec![ColorOrQuery::Color(self.fg)]
            ),
            Osc::ChangeDynamicColors(
                DynamicColorNumber::TextBackgroundColor,
                vec![ColorOrQuery::Color(self.bg)]
            )
        )?;
        backend.flush()?;
        Ok(())
    }
}

/// Background worker for "check for updates now" / auto-update polling in
/// the TUI. Same background-thread-plus-channel shape used elsewhere in
/// this file for `VoiceManager`.
enum UpdateCommand {
    Check,
    UpdateAndRestart,
}

struct UpdateManagerTui {
    auto_update_enabled: bool,
    status: String,
    tx_cmd: Option<std::sync::mpsc::Sender<UpdateCommand>>,
    rx_event: Option<std::sync::mpsc::Receiver<String>>,
}

impl UpdateManagerTui {
    fn new(auto_update_enabled: bool) -> Self {
        Self {
            auto_update_enabled,
            status: "Update: idle".to_string(),
            tx_cmd: None,
            rx_event: None,
        }
    }

    fn start_worker_if_needed(&mut self) {
        if self.tx_cmd.is_some() {
            return;
        }
        let (tx_cmd, rx_cmd) = std::sync::mpsc::channel::<UpdateCommand>();
        let (tx_event, rx_event) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            while let Ok(command) = rx_cmd.recv() {
                let current_version =
                    semver::Version::parse(env!("CARGO_PKG_VERSION"))
                        .unwrap_or_else(|_| semver::Version::new(0, 0, 0));
                let owner = "she-workss";
                let repo = "stealcode";
                let token = std::env::var("STEALCODE_GH_TOKEN").ok();
                let channel = release_channel::ReleaseChannel::current();
                match command {
                    UpdateCommand::Check => {
                        let message = match auto_update::check_now_blocking(
                            owner,
                            repo,
                            token,
                            current_version,
                            channel,
                        ) {
                            Ok(Some(version)) => {
                                format!("Update available: v{version}")
                            }
                            Ok(None) => "Up to date".to_string(),
                            Err(error) => {
                                format!("Check failed: {error}")
                            }
                        };
                        let _ = tx_event.send(message);
                    }
                    UpdateCommand::UpdateAndRestart => {
                        let _ =
                            tx_event.send("Downloading update...".to_string());
                        match auto_update::update_now_blocking(
                            owner,
                            repo,
                            token,
                            current_version,
                            channel,
                        ) {
                            Ok(version) => {
                                let _ = tx_event.send(format!(
                                    "Update installed: v{version} \
                                     - restarting"
                                ));
                                if let Err(error) =
                                    auto_update::restart_updated_app()
                                {
                                    let _ = tx_event.send(format!(
                                        "Restart failed: {error}"
                                    ));
                                }
                            }
                            Err(error) => {
                                let _ = tx_event
                                    .send(format!("Update failed: {error}"));
                            }
                        }
                    }
                }
            }
        });
        self.tx_cmd = Some(tx_cmd);
        self.rx_event = Some(rx_event);
    }

    fn check_now(&mut self) {
        self.start_worker_if_needed();
        self.status = "Checking...".to_string();
        if let Some(tx) = &self.tx_cmd {
            let _ = tx.send(UpdateCommand::Check);
        }
    }

    fn update_and_restart(&mut self) {
        self.start_worker_if_needed();
        self.status = "Downloading update...".to_string();
        if let Some(tx) = &self.tx_cmd {
            let _ = tx.send(UpdateCommand::UpdateAndRestart);
        }
    }

    fn toggle(&mut self) {
        self.auto_update_enabled = !self.auto_update_enabled;
    }

    fn poll_events(&mut self) {
        if let Some(rx) = &self.rx_event {
            if let Ok(message) = rx.try_recv() {
                self.status = message;
            }
        }
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
    button_rects: Vec<(SoundName, Rect)>,
    last_played: Option<SoundName>,
    #[cfg(feature = "voice")]
    voice: VoiceManager,
    updates: UpdateManagerTui,
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
            button_rects: Vec::new(),
            last_played: None,
            #[cfg(feature = "voice")]
            voice: VoiceManager::new(),
            updates: UpdateManagerTui::new(true),
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

fn rect_contains(rect: Rect, x: u16, y: u16) -> bool {
    x >= rect.x
        && x < rect.x + rect.width
        && y >= rect.y
        && y < rect.y + rect.height
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
                state.mode_label()
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
            state.action_label = "Ctrl+P/F/H/T/V/U/?/C".into();
        }
        Action::TogglePushToTalk => {
            #[cfg(feature = "voice")]
            {
                state.voice.toggle();
                state.action_label = if state.voice.is_recording {
                    "PTT: Recording...".into()
                } else {
                    "PTT: Processing...".into()
                };
            }
        }
        Action::CheckForUpdates => {
            state.updates.check_now();
            state.action_label = "Checking for updates...".into();
        }
        Action::ToggleAutoUpdate => {
            state.updates.toggle();
            state.action_label = format!(
                "Auto-update: {}",
                if state.updates.auto_update_enabled {
                    "ON"
                } else {
                    "OFF"
                }
            );
        }
        Action::UpdateAndRestart => {
            state.updates.update_and_restart();
            state.action_label = "Updating and restarting...".into();
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
        decset!(SGRMouse)
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
        decreset!(AnyEventMouse)
    )?;
    backend.flush()?;
    Ok(())
}

pub fn run_tui(
    _settings: &Settings,
    project: Option<&Path>,
) -> anyhow::Result<()> {
    let (mut terminal, events) = init_terminal()?;
    let color_mode = detect_color_mode(&mut terminal, &events)?;
    let mut state = AppState::new(color_mode);
    if color_mode.supports_osc() {
        state.scheme().apply_osc(&mut terminal)?;
    }
    // If a previous session staged an update (e.g. via `stealcode upgrade`
    // or an in-app update while another instance was running), apply it now:
    // the helper swaps the binary in and relaunches this process as the new
    // version; exit the stale instance.
    #[cfg(target_os = "windows")]
    if auto_update::apply_staged_update_on_startup_blocking()? {
        return Ok(());
    }
    // Remove stale update/install/old dirs from a crashed previous update,
    // at startup (only empty dirs are removed, see `cleanup_windows`).
    #[cfg(target_os = "windows")]
    if let Err(error) = auto_update::cleanup_windows_blocking() {
        eprintln!("failed to clean up update dirs: {error}");
    }
    let result = run(&mut terminal, &events, &mut state, project);
    restore_terminal(&mut terminal, color_mode)?;
    // If a silent update finished while we were running, spawn the helper
    // to apply it as we exit (renaming a running exe is legal on Windows,
    // so the helper's retry loop rides out our shutdown).
    #[cfg(target_os = "windows")]
    auto_update::finalize_auto_update_on_quit_blocking();
    result
}

fn run(
    terminal: &mut AppTerminal,
    events: &EventReader,
    state: &mut AppState,
    _project: Option<&Path>,
) -> Result<()> {
    loop {
        terminal.draw(|f| render(f, state))?;
        #[cfg(feature = "voice")]
        state.voice.poll_events();
        state.updates.poll_events();
        if events.poll(Some(Duration::from_millis(100)), |_| true)? {
            let event = events.read(|_| true)?;
            if let Event::Mouse(mouse) = &event {
                state.event_raw = format!("{mouse:?}");
                state.mouse_pos = Some((mouse.column, mouse.row));
                if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                {
                    let hit = state
                        .button_rects
                        .iter()
                        .find(|(_, rect)| {
                            rect_contains(*rect, mouse.column, mouse.row)
                        })
                        .map(|(sound, _)| *sound);
                    if let Some(sound) = hit {
                        engine::play(sound);
                        state.last_played = Some(sound);
                        state.action_label =
                            format!("Played: {}", sound.label());
                    }
                } else {
                    state.action_label = format!(
                        "Mouse: kind={:?} col={} row={}",
                        mouse.kind, mouse.column, mouse.row
                    );
                }
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
    }
    Ok(())
}

fn render(f: &mut Frame<'_>, state: &mut AppState) {
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
            state.mode_label()
        ),
        s.block_style(),
        s,
    );
    let mouse = match state.mouse_pos {
        Some((x, y)) => format!("Mouse: x={x} y={y}"),
        None => "Mouse: -".into(),
    };
    render_panel(f, chunks[3], "Mouse", &mouse, s.accent_style(), s);
    #[cfg(feature = "voice")]
    {
        let voice_content = format!(
            "[{}] {}\n{}",
            if state.voice.is_recording {
                "●"
            } else {
                "○"
            },
            state.voice.status,
            if state.voice.text.is_empty() {
                " "
            } else {
                &state.voice.text
            }
        );
        render_panel(
            f,
            chunks[4],
            "Voice Input (Ctrl+G)",
            &voice_content,
            if state.voice.is_recording {
                s.accent_style()
            } else {
                s.block_style()
            },
            s,
        );
    }
    let update_content = format!(
        "Auto-update: {}  {}",
        if state.updates.auto_update_enabled {
            "ON"
        } else {
            "OFF"
        },
        state.updates.status,
    );
    render_panel(
        f,
        chunks[5],
        "Update (Ctrl+U check, Ctrl+A toggle, Ctrl+R update)",
        &update_content,
        s.block_style(),
        s,
    );
    render_sound_buttons(f, chunks[6], state, s);
    let hint = if cfg!(feature = "voice") {
        format!(
            "Ctrl+P (Palette) Ctrl+F (Search) Ctrl+T (Title++) Tab (Next Theme) Ctrl+G (Voice) Ctrl+U (Update) Ctrl+H (Help) Ctrl+C (Exit) | {}",
            state.mode_label()
        )
    } else {
        format!(
            "Ctrl+P (Palette) Ctrl+F (Search) Ctrl+T (Title++) Tab (Next Theme) Ctrl+U (Update) Ctrl+H (Help) Ctrl+C (Exit) | {}",
            state.mode_label()
        )
    };
    f.render_widget(Paragraph::new(hint).style(s.hint_style()), chunks[7]);
}

fn render_sound_buttons(
    f: &mut Frame<'_>,
    area: Rect,
    state: &mut AppState,
    s: ColorScheme,
) {
    let sounds = SoundName::ALL;
    let constraints =
        vec![Constraint::Ratio(1, sounds.len() as u32); sounds.len()];
    let slots = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);
    state.button_rects.clear();
    for (sound, rect) in sounds.into_iter().zip(slots.iter()) {
        let is_active = state.last_played == Some(sound);
        let style = if is_active {
            s.accent_style()
        } else {
            s.block_style()
        };
        let border_style = if is_active {
            s.accent_style()
        } else {
            s.border_style()
        };
        let block = Block::bordered().border_style(border_style);
        f.render_widget(
            Paragraph::new(sound.label())
                .alignment(Alignment::Center)
                .style(style)
                .block(block),
            *rect,
        );
        state.button_rects.push((sound, *rect));
    }
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
            Constraint::Length(5),
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
