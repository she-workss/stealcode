use std::io::Write as _;

use anyhow::Result;
use ratatui::{
    Frame, Terminal,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style, Stylize},
    widgets::{Block, Paragraph},
};
use ratatui_termina::{
    TerminaBackend,
    termina::{
        EventReader, PlatformTerminal, Terminal as _,
        escape::{
            csi::{self, Csi},
            osc::{DynamicColorNumber, Osc},
        },
        event::{Event, KeyCode, KeyEvent, KeyEventKind, Modifiers},
        style::RgbColor,
    },
};
use settings::{self, Settings};

type AppTerminal = Terminal<TerminaBackend<PlatformTerminal>>;

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

fn is_hotkey(event: &KeyEvent, modifier: Modifiers, key: char) -> bool {
    event.modifiers == modifier
        && matches!(event.code, KeyCode::Char(c) if c == key)
}

struct AppState {
    last_raw_event: String,
    triggered_action: String,
    title: String,
    fg: RgbColor,
    bg: RgbColor,
    mouse_pos: Option<(u32, u32)>,
}

impl AppState {
    fn new() -> Self {
        Self {
            last_raw_event: String::from("Press any key"),
            triggered_action: String::from("-"),
            title: String::from("ratatui + termina"),
            fg: RgbColor::new(255, 255, 255),
            bg: RgbColor::new(0, 0, 0),
            mouse_pos: None,
        }
    }
}

pub fn run_tui(_settings: &Settings) -> Result<()> {
    let (mut terminal, events) = init_terminal()?;
    run(&mut terminal, &events)?;
    restore_terminal(&mut terminal)?;
    Ok(())
}

fn init_terminal() -> Result<(AppTerminal, EventReader)> {
    let mut output = PlatformTerminal::new()?;
    output.enter_raw_mode()?;
    write!(
        output,
        "{}{}{}{}{}{}{}{}{}",
        decset!(ClearAndEnableAlternateScreen),
        decreset!(ShowCursor),
        Csi::Window(Box::new(csi::Window::PushIconAndWindowTitle)),
        Osc::SetIconNameAndWindowTitle("ratatui + termina"),
        decset!(MouseTracking),
        decset!(ButtonEventMouse),
        decset!(AnyEventMouse),
        decset!(RXVTMouse),
        decset!(SGRMouse),
    )?;
    output.flush()?;
    let events = output.event_reader();
    let backend = TerminaBackend::new(output);
    let terminal = Terminal::new(backend)?;
    Ok((terminal, events))
}

fn restore_terminal(terminal: &mut AppTerminal) -> Result<()> {
    let backend = terminal.backend_mut();
    write!(
        backend,
        "{}{}{}{}{}{}{}{}",
        decset!(ShowCursor),
        decreset!(ClearAndEnableAlternateScreen),
        Csi::Window(Box::new(csi::Window::PopIconAndWindowTitle)),
        Osc::ResetDynamicColor(DynamicColorNumber::TextForegroundColor),
        Osc::ResetDynamicColor(DynamicColorNumber::TextBackgroundColor),
        decreset!(MouseTracking),
        decreset!(ButtonEventMouse),
        decreset!(AnyEventMouse),
    )?;
    backend.flush()?;
    Ok(())
}

fn apply_colors(terminal: &mut AppTerminal, state: &AppState) -> Result<()> {
    let backend = terminal.backend_mut();
    write!(
        backend,
        "{}{}",
        Osc::ChangeDynamicColors(
            DynamicColorNumber::TextForegroundColor,
            vec![state.fg.into()],
        ),
        Osc::ChangeDynamicColors(
            DynamicColorNumber::TextBackgroundColor,
            vec![state.bg.into()],
        ),
    )?;
    backend.flush()?;
    Ok(())
}

fn apply_title(terminal: &mut AppTerminal, state: &AppState) -> Result<()> {
    let backend = terminal.backend_mut();
    write!(backend, "{}", Osc::SetIconNameAndWindowTitle(&state.title),)?;
    backend.flush()?;
    Ok(())
}

fn run(terminal: &mut AppTerminal, events: &EventReader) -> Result<()> {
    let mut state = AppState::new();
    loop {
        terminal.draw(|f| render(f, &state))?;
        let event = events.read(|_| true)?;
        match &event {
            Event::Key(key) => {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                state.last_raw_event = format!("{key:?}");
                if is_hotkey(key, Modifiers::CONTROL, 'c') {
                    break;
                } else if is_hotkey(key, Modifiers::CONTROL, 'p') {
                    state.triggered_action = "OpenCommandPalette".to_string();
                } else if is_hotkey(key, Modifiers::CONTROL, 'f') {
                    state.triggered_action = "Search".to_string();
                } else if is_hotkey(key, Modifiers::CONTROL, 't') {
                    state.title = format!(
                        "ratatui + termina [{}]",
                        chrono_or_counter(&mut state.triggered_action)
                    );
                    apply_title(terminal, &state)?;
                    state.triggered_action =
                        format!("Title: \"{}\"", state.title);
                } else if is_hotkey(key, Modifiers::CONTROL, 'v') {
                    state.fg = RgbColor::new(255, 255, 255);
                    state.bg = RgbColor::new(0, 0, 0);
                    apply_colors(terminal, &state)?;
                    state.triggered_action =
                        "Colors: black and white".to_string();
                } else if is_hotkey(key, Modifiers::CONTROL, 'b') {
                    state.fg = RgbColor::new(128, 128, 255);
                    state.bg = RgbColor::new(0, 64, 0);
                    apply_colors(terminal, &state)?;
                    state.triggered_action =
                        "Colors: blue and dark green".to_string();
                } else if is_hotkey(key, Modifiers::CONTROL, 'n') {
                    state.fg = RgbColor::new(255, 220, 100);
                    state.bg = RgbColor::new(40, 0, 60);
                    apply_colors(terminal, &state)?;
                    state.triggered_action =
                        "Colors: yellow and purple".to_string();
                } else {
                    state.triggered_action = format!("Unknown: {event:?}");
                }
            }
            Event::Mouse(mouse) => {
                state.last_raw_event = format!("{mouse:?}");
                state.mouse_pos = Some((mouse.column as u32, mouse.row as u32));
                state.triggered_action = format!(
                    "Mouse: kind={:?} col={} row={}",
                    mouse.kind, mouse.column, mouse.row,
                );
            }
            other => {
                state.last_raw_event = format!("{other:?}");
            }
        }
    }
    Ok(())
}

static COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

fn chrono_or_counter(_: &mut String) -> String {
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("#{n}")
}

fn render(f: &mut Frame<'_>, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(f.area());

    f.render_widget(
        Paragraph::new(format!("Event: {}", state.last_raw_event))
            .block(Block::bordered().title("Termina Event")),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(format!("Action: {}", state.triggered_action))
            .style(Style::default().fg(Color::Green).bold())
            .block(Block::bordered().title("Hotkey")),
        chunks[1],
    );

    f.render_widget(
        Paragraph::new(format!(
            "Title: \"{}\" Fg: {:?} Bg: {:?}",
            state.title, state.fg, state.bg,
        ))
        .block(Block::bordered().title("State")),
        chunks[2],
    );

    let mouse_text = match state.mouse_pos {
        Some((x, y)) => format!("Mouse: x={x} y={y}"),
        None => "Mouse: -".to_string(),
    };
    f.render_widget(
        Paragraph::new(mouse_text)
            .style(Style::default().fg(Color::Cyan))
            .block(Block::bordered().title("Mouse")),
        chunks[3],
    );

    f.render_widget(
        Paragraph::new(
            "Ctrl+P (Palette), Ctrl+F (Search), Ctrl+T (Title++), \
             Ctrl+V (B/W), Ctrl+B (Blue/Green), Ctrl+N (Yellow/Purple), Ctrl+C (Exit)",
        )
        .dark_gray(),
        chunks[4],
    );
}
