//! Interactive viewer: pick any ttfx effect and watch it run on the wordmark.
//!
//! Run from the crate root:  cargo run --example all
//!
//! Controls:
//! - Up/Down or j/k: pick an effect; the preview restarts with it.
//! - Enter/Space: restart the selected effect (finished effects also
//!   auto-restart after a short pause).
//! - q / Esc / Ctrl+C: quit.
//!
//! Each frame from the engine is rendered into the ratatui buffer through
//! `Terminal::frame_cells`, which decodes the ANSI output string into styled
//! cells - the integration point this example demonstrates.

use std::{
    io,
    io::Write as _,
    time::{Duration, Instant},
};

use ratatui::{
    Frame, Terminal,
    backend::TerminaBackend,
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color as TuiColor, Modifier, Style},
    termina::{
        PlatformTerminal, Terminal as _,
        escape::csi::{self, Csi},
        event::{Event, KeyCode, KeyEventKind, Modifiers},
    },
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};
use ttfx::{
    engine::{
        animation::ExistingColorHandling,
        canvas::Anchor,
        ctx::{Clock, EngineCtx},
        effect::Effect,
        terminal::{FrameCell, TerminalConfig},
    },
    utils::{ansi::ColorCode, graphics::Color, rng::Rng},
};

/// The preview canvas is fixed so the rendered grid never depends on the user's
/// terminal size, and the effect has room to move around the text.
const PREVIEW_WIDTH: i64 = 46;
const PREVIEW_HEIGHT: i64 = 6;
/// How long a finished effect keeps its final frame before replaying.
const RESTART_DELAY: Duration = Duration::from_secs(4);

/// The wordmark, four rows tall. Each letter occupies four columns followed by
/// one space; the word reads "StealCode" left to right.
pub const WORDMARK: &[&str; 4] = &[
    "     ▄                                ▄     ",
    "█▀▀▀ █▀▀  █▀▀█ ▀▀▀█ █    █▀▀▀ █▀▀█ █▀▀█ █▀▀█",
    "▀▀▀█ █    █▀▀▀ █▀▀█ █    █    █  █ █  █ █▀▀▀",
    "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀",
];

/// Per-letter colors: "Steal" is gray, "C" red, "o" yellow, "d" green and the
/// last "e" blue.
pub const LETTER_COLORS: &[Color] = &[
    Color::DarkGray,
    Color::DarkGray,
    Color::DarkGray,
    Color::DarkGray,
    Color::DarkGray,
    Color::Red,
    Color::Yellow,
    Color::Green,
    Color::Blue,
];

/// How many columns each letter occupies (four glyphs plus one separator).
const LETTER_WIDTH: usize = 5;

/// The wordmark joined into a single input string with each letter wrapped in
/// an SGR foreground sequence (`\x1b[38;5;<xterm>m ... \x1b[0m`). Columns that
/// belong to no letter (leading/trailing padding) stay plain.
pub fn input_text() -> String {
    let mut out = String::new();
    for (row, line) in WORDMARK.iter().enumerate() {
        if row > 0 {
            out.push('\n');
        }
        let chars: Vec<char> = line.chars().collect();
        for (letter, color) in LETTER_COLORS.iter().enumerate() {
            let start = letter * LETTER_WIDTH;
            let end = (start + 4).min(chars.len());
            if start >= chars.len() {
                break;
            }
            let code = color.xterm_color.expect("named colors are xterm");
            if chars[start..end].iter().any(|c| *c != ' ') {
                out.push_str(&format!("\x1b[38;5;{code}m"));
            }
            for c in &chars[start..end] {
                out.push(*c);
            }
            if chars[start..end].iter().any(|c| *c != ' ') {
                out.push_str("\x1b[0m");
            }
            if end < chars.len() {
                out.push(chars[end]);
            }
        }
    }
    out
}

/// DECSET/DECRST ?1049: switch to (or back from) the alternate screen so the
/// viewer owns the whole terminal and no scrolled-back content bleeds through.
/// Mirrors `ClearAndEnableAlternateScreen` handling in ratatui_termina.rs.
fn alternate_screen(enable: bool) -> Csi {
    let mode = csi::DecPrivateMode::Code(
        csi::DecPrivateModeCode::ClearAndEnableAlternateScreen,
    );
    Csi::Mode(if enable {
        csi::Mode::SetDecPrivateMode(mode)
    } else {
        csi::Mode::ResetDecPrivateMode(mode)
    })
}

fn preview_config() -> TerminalConfig {
    TerminalConfig {
        canvas_width: PREVIEW_WIDTH,
        canvas_height: PREVIEW_HEIGHT,
        ignore_terminal_dimensions: true,
        anchor_text: Anchor::C,
        frame_rate: 60,
        existing_color_handling: ExistingColorHandling::Dynamic,
        ..Default::default()
    }
}

/// A running effect: the engine plus the latest rendered cells.
struct Preview {
    ctx: EngineCtx,
    effect: Box<dyn Effect>,
    cells: Vec<Vec<FrameCell>>,
    frame: u64,
    done: bool,
    done_at: Instant,
}

impl Preview {
    fn new(name: &str, input: &str) -> Result<Self, String> {
        let mut ctx = EngineCtx::new(
            input,
            preview_config(),
            Rng::from_entropy(),
            Clock::real(),
        )
        .map_err(|e| e.to_string())?;
        let mut effect = ttfx::effects::build_effect(name)
            .ok_or_else(|| format!("unknown effect: {name}"))?;
        effect.build(&mut ctx).map_err(|e| e.to_string())?;
        let cells = ctx.terminal.frame_cells();
        Ok(Preview {
            ctx,
            effect,
            cells,
            frame: 0,
            done: false,
            done_at: Instant::now(),
        })
    }

    /// Advance one frame. `next_frame` paces itself (frame_rate sleeps), so
    /// this also throttles the whole UI loop while an effect is running.
    fn tick(&mut self) {
        if self.done {
            return;
        }
        if self.effect.next_frame(&mut self.ctx).is_some() {
            self.frame += 1;
            self.cells = self.ctx.terminal.frame_cells();
        } else {
            self.done = true;
            self.done_at = Instant::now();
        }
    }
}

struct App {
    names: Vec<String>,
    selected: usize,
    input: String,
    preview: Option<Preview>,
    message: Option<String>,
    list_state: ListState,
}

impl App {
    fn new(input: String) -> Self {
        let names = ttfx::effects::effect_names();
        let mut app = App {
            names,
            selected: 0,
            input,
            preview: None,
            message: None,
            list_state: ListState::default(),
        };
        app.select(0);
        app
    }

    fn select(&mut self, index: usize) {
        let len = self.names.len();
        self.selected = if len == 0 { 0 } else { index % len };
        self.list_state.select(Some(self.selected));
        self.rebuild();
    }

    fn rebuild(&mut self) {
        self.message = None;
        match Preview::new(&self.names[self.selected], &self.input) {
            Ok(preview) => self.preview = Some(preview),
            Err(e) => {
                self.message = Some(e);
                self.preview = None;
            }
        }
    }

    fn tick(&mut self) {
        if let Some(preview) = &mut self.preview {
            preview.tick();
            if preview.done && preview.done_at.elapsed() >= RESTART_DELAY {
                self.rebuild();
            }
        }
    }

    /// Returns true when the key should quit the viewer.
    fn handle(&mut self, code: &KeyCode) -> bool {
        match code {
            KeyCode::Char('q') | KeyCode::Escape => true,
            KeyCode::Char('j') | KeyCode::Down => {
                self.select(self.selected + 1);
                false
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.select(self.selected.wrapping_sub(1));
                false
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                self.rebuild();
                false
            }
            _ => false,
        }
    }
}

fn ui(app: &mut App, frame: &mut Frame) {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(1)])
        .split(frame.area());
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(28), Constraint::Percentage(72)])
        .split(vertical[0]);

    let items: Vec<ListItem> = app
        .names
        .iter()
        .map(|name| ListItem::new(name.as_str()))
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Effects"))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, horizontal[0], &mut app.list_state);

    let title = match &app.preview {
        Some(preview) => {
            let mut title = format!(
                "{}  [frame {}]",
                app.names[app.selected], preview.frame
            );
            if preview.done {
                title.push_str("  (done)");
            }
            title
        }
        None => "preview".to_string(),
    };
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(horizontal[1]);
    frame.render_widget(block, horizontal[1]);
    let cells = app
        .preview
        .as_ref()
        .map(|preview| preview.cells.as_slice())
        .unwrap_or(&[]);
    frame.render_widget(PreviewWidget { cells }, inner);

    let footer = match &app.message {
        Some(message) => message.clone(),
        None => "j/k or arrows: pick effect   Enter/Space: restart   q/Esc/Ctrl+C: quit".to_string(),
    };
    frame.render_widget(Paragraph::new(footer), vertical[1]);
}

/// Draws the engine's frame cells into a rect, centered, wiping everything
/// else in the rect first so nothing lingers between frames or effects.
struct PreviewWidget<'a> {
    cells: &'a [Vec<FrameCell>],
}

impl ratatui::widgets::Widget for PreviewWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                buf[(x, y)].reset();
            }
        }
        let Some(width) = self.cells.first().map(|row| row.len() as u16) else {
            return;
        };
        let height = self.cells.len() as u16;
        let x0 = area.left() + area.width.saturating_sub(width) / 2;
        let y0 = area.top() + area.height.saturating_sub(height) / 2;
        for (row_index, row) in self.cells.iter().enumerate() {
            // Engine rows run bottom-up; screen rows run top-down.
            let y = y0 + height - 1 - row_index as u16;
            if y < area.top() || y >= area.bottom() {
                continue;
            }
            for (column_index, cell) in row.iter().enumerate() {
                let x = x0 + column_index as u16;
                if x < area.left() || x >= area.right() {
                    continue;
                }
                buf[(x, y)]
                    .set_symbol(&cell.symbol)
                    .set_style(style_of(cell));
            }
        }
    }
}

fn style_of(cell: &FrameCell) -> Style {
    let mut style = Style::default();
    if let Some(fg) = &cell.fg {
        style = style.fg(to_tui_color(fg));
    }
    if let Some(bg) = &cell.bg {
        style = style.bg(to_tui_color(bg));
    }
    if cell.bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    if cell.italic {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if cell.underline {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    if cell.blink {
        style = style.add_modifier(Modifier::SLOW_BLINK);
    }
    if cell.reverse {
        style = style.add_modifier(Modifier::REVERSED);
    }
    if cell.hidden {
        style = style.add_modifier(Modifier::HIDDEN);
    }
    if cell.strike {
        style = style.add_modifier(Modifier::CROSSED_OUT);
    }
    style
}

fn to_tui_color(code: &ColorCode) -> TuiColor {
    match code {
        ColorCode::Rgb(hex) => {
            let hex = hex.trim_start_matches('#');
            if hex.len() < 6 {
                return TuiColor::Reset;
            }
            let component = |index: usize| {
                u8::from_str_radix(&hex[index..index + 2], 16).unwrap_or(0)
            };
            TuiColor::Rgb(component(0), component(2), component(4))
        }
        ColorCode::Xterm(n) => TuiColor::Indexed(*n),
    }
}

/// Owns the ratatui terminal; on drop leaves the alternate screen and restores
/// cooked mode so a panic or an early exit cannot leave the shell in raw mode.
struct Tui {
    terminal: Terminal<TerminaBackend<PlatformTerminal>>,
}

impl Drop for Tui {
    fn drop(&mut self) {
        let backend = self.terminal.backend_mut();
        let show_cursor = Csi::Mode(csi::Mode::SetDecPrivateMode(
            csi::DecPrivateMode::Code(csi::DecPrivateModeCode::ShowCursor),
        ));
        let _ = write!(backend, "{}{}", show_cursor, alternate_screen(false));
        let _ = backend.flush();
        let _ = backend.terminal_mut().enter_cooked_mode();
    }
}

fn main() -> io::Result<()> {
    let mut output = PlatformTerminal::new()?;
    output.enter_raw_mode()?;
    write!(output, "{}", alternate_screen(true))?;
    output.flush()?;
    let reader = output.event_reader();
    let mut tui = Tui {
        terminal: Terminal::new(TerminaBackend::new(output))?,
    };
    tui.terminal.hide_cursor()?;

    let mut app = App::new(input_text());

    loop {
        app.tick();

        tui.terminal.draw(|frame| ui(&mut app, frame))?;

        if reader.poll(Some(Duration::from_millis(10)), |_| true)? {
            let event = reader.read(|_| true)?;
            if let Event::Key(key) = event {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if key.code == KeyCode::Char('c')
                    && key.modifiers.contains(Modifiers::CONTROL)
                {
                    break;
                }
                if app.handle(&key.code) {
                    break;
                }
            }
        }
    }

    tui.terminal.show_cursor()?;
    Ok(())
}
