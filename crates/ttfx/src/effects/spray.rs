//! spray, ported from effects/effect_spray.py.

use rustc_hash::FxHashMap;

use crate::{
    effects::common::{
        parse_color, parse_easing, parse_gradient_direction,
        parse_positive_float_range,
    },
    engine::{
        animation::{ExistingColorHandling, VisualParams},
        character::CharId,
        ctx::{EffectHooks, EngineCtx},
        effect::Effect,
        error::EngineError,
        events::{CallerKey, Event, EventAction},
        terminal::{CharacterFilter, CharacterSort},
    },
    utils::{
        easing::Easing,
        geometry::Coord,
        graphics::{Color, ColorPair, Gradient, GradientDirection},
        pycompat::floor_div,
    },
};

/// SprayIterator.SprayPosition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprayPosition {
    N,
    Ne,
    E,
    Se,
    S,
    Sw,
    W,
    Nw,
    Center,
}

fn parse_spray_position(s: &str) -> Result<SprayPosition, String> {
    Ok(match s {
        "n" => SprayPosition::N,
        "ne" => SprayPosition::Ne,
        "e" => SprayPosition::E,
        "se" => SprayPosition::Se,
        "s" => SprayPosition::S,
        "sw" => SprayPosition::Sw,
        "w" => SprayPosition::W,
        "nw" => SprayPosition::Nw,
        "center" => SprayPosition::Center,
        _ => {
            return Err(format!(
                "invalid choice: '{s}' (choose from 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw', 'center')"
            ));
        }
    })
}

#[derive(Debug, Clone)]
pub struct SprayConfig {
    /// Position for the spray origin.
    pub spray_position: SprayPosition,

    /// Number of characters to spray per tick as a percent of the total number
    /// of characters.
    pub spray_volume: f64,

    /// Movement speed range of the characters.
    pub movement_speed_range: (f64, f64),

    /// Easing function to use for character movement.
    pub movement_easing: Easing,

    /// Space separated, unquoted, list of colors for the final color gradient.
    pub final_gradient_stops: Vec<Color>,

    /// Number of gradient steps to use.
    pub final_gradient_steps: Vec<i64>,

    /// Direction of the final gradient.
    pub final_gradient_direction: GradientDirection,
}

impl Default for SprayConfig {
    fn default() -> Self {
        Self {
            spray_position: parse_spray_position("e").expect("valid literal"),
            spray_volume: 0.005,
            movement_speed_range: parse_positive_float_range("0.6-1.4")
                .expect("valid literal"),
            movement_easing: parse_easing("out_expo").expect("valid literal"),
            final_gradient_stops: vec![
                parse_color("8A008A").expect("valid color"),
                parse_color("00D1FF").expect("valid color"),
                parse_color("FFFFFF").expect("valid color"),
            ],
            final_gradient_steps: vec![12],
            final_gradient_direction: parse_gradient_direction("vertical")
                .expect("valid literal"),
        }
    }
}

#[derive(Debug)]
pub struct Spray {
    config: SprayConfig,
    pending_chars: Vec<CharId>,
    character_final_color_map: FxHashMap<CharId, ColorPair>,
    volume: i64,
}

impl Spray {
    pub fn new(config: SprayConfig) -> Self {
        Spray {
            config,
            pending_chars: Vec::new(),
            character_final_color_map: FxHashMap::default(),
            volume: 1,
        }
    }
}

impl EffectHooks for Spray {}
impl Effect for Spray {
    fn build(&mut self, ctx: &mut EngineCtx) -> Result<(), EngineError> {
        let final_gradient = Gradient::new(
            &self.config.final_gradient_stops,
            &self.config.final_gradient_steps,
            false,
            false,
        )
        .map_err(EngineError::Other)?;
        let final_gradient_mapping = final_gradient
            .build_coordinate_color_mapping(
                ctx.terminal.canvas.text_bottom,
                ctx.terminal.canvas.text_top,
                ctx.terminal.canvas.text_left,
                ctx.terminal.canvas.text_right,
                self.config.final_gradient_direction,
            )
            .map_err(EngineError::Other)?;

        let dynamic = ctx.terminal.config.existing_color_handling
            == ExistingColorHandling::Dynamic;
        let characters = {
            let filter = CharacterFilter::default();
            ctx.terminal.get_characters(
                &mut ctx.rng,
                filter,
                CharacterSort::TopToBottomLeftToRight,
            )
        };
        for &id in &characters {
            let final_colors = {
                let ch = &ctx.terminal.arena[id.0 as usize];
                if dynamic {
                    ColorPair::new(
                        ch.animation.input_fg_color,
                        ch.animation.input_bg_color,
                    )
                } else {
                    ColorPair::new(
                        Some(
                            *final_gradient_mapping
                                .get(&ch.input_coord)
                                .unwrap(),
                        ),
                        None,
                    )
                }
            };
            self.character_final_color_map.insert(id, final_colors);
        }

        let canvas = &ctx.terminal.canvas;
        let spray_origin = match self.config.spray_position {
            SprayPosition::Center => canvas.center,
            SprayPosition::N => {
                Coord::new(floor_div(canvas.right, 2), canvas.top)
            }
            SprayPosition::Nw => Coord::new(canvas.left, canvas.top),
            SprayPosition::W => {
                Coord::new(canvas.left, floor_div(canvas.top, 2))
            }
            SprayPosition::Sw => Coord::new(canvas.left, canvas.bottom),
            SprayPosition::S => {
                Coord::new(floor_div(canvas.right, 2), canvas.bottom)
            }
            SprayPosition::Se => Coord::new(canvas.right - 1, canvas.bottom),
            SprayPosition::E => {
                Coord::new(canvas.right - 1, floor_div(canvas.top, 2))
            }
            SprayPosition::Ne => Coord::new(canvas.right - 1, canvas.top),
        };

        for id in characters {
            let (input_coord, input_symbol, uses_pre) = {
                let ch = &ctx.terminal.arena[id.0 as usize];
                (
                    ch.input_coord,
                    ch.input_symbol.clone(),
                    ch.uses_input_preexisting_colors,
                )
            };
            let speed = ctx.rng.uniform(
                self.config.movement_speed_range.0,
                self.config.movement_speed_range.1,
            );
            let input_coord_path = {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.motion.set_coordinate(spray_origin);
                let path_id = ch
                    .motion
                    .new_path(
                        speed,
                        Some(self.config.movement_easing),
                        None,
                        0,
                        false,
                        "",
                    )
                    .map_err(EngineError::Other)?;
                ch.motion
                    .paths
                    .get_mut(&path_id)
                    .unwrap()
                    .new_waypoint(input_coord, None, "")
                    .map_err(EngineError::Other)?;
                path_id
            };
            ctx.register_event(
                id,
                Event::PathActivated,
                CallerKey::Path(input_coord_path.clone()),
                EventAction::SetLayer(1),
            )
            .map_err(EngineError::Other)?;
            ctx.register_event(
                id,
                Event::PathComplete,
                CallerKey::Path(input_coord_path.clone()),
                EventAction::SetLayer(0),
            )
            .map_err(EngineError::Other)?;

            let droplet_scn = {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.animation.new_scene(false, None, None, "", uses_pre)
            };
            let final_colors =
                *self.character_final_color_map.get(&id).unwrap();
            if dynamic {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                let scene = ch.animation.scenes.get_mut(&droplet_scn).unwrap();
                for _ in 0..7 {
                    scene
                        .add_frame(
                            &input_symbol,
                            20,
                            VisualParams {
                                colors: Some(final_colors),
                                ..Default::default()
                            },
                        )
                        .map_err(EngineError::Other)?;
                }
            } else {
                let start_color = *ctx.rng.choice(&final_gradient.spectrum);
                let final_fg =
                    final_colors.fg_color.expect("gradient mapping fg");
                let spray_gradient =
                    Gradient::with_steps(&[start_color, final_fg], 7, false)
                        .map_err(EngineError::Other)?;
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.animation
                    .scenes
                    .get_mut(&droplet_scn)
                    .unwrap()
                    .apply_gradient_to_symbols(
                        std::slice::from_ref(&input_symbol),
                        20,
                        Some(&spray_gradient),
                        None,
                    )
                    .map_err(EngineError::Other)?;
            }
            ctx.activate_scene(self, id, &droplet_scn);
            ctx.activate_path(self, id, &input_coord_path);
            self.pending_chars.push(id);
        }
        ctx.rng.shuffle(&mut self.pending_chars);
        self.volume = std::cmp::max(
            (self.pending_chars.len() as f64 * self.config.spray_volume) as i64,
            1,
        );
        Ok(())
    }

    fn next_frame(&mut self, ctx: &mut EngineCtx) -> Option<String> {
        if !self.pending_chars.is_empty() || !ctx.active_characters.is_empty() {
            if !self.pending_chars.is_empty() {
                for _ in 0..ctx.rng.randint(1, self.volume) {
                    if let Some(next_character) = self.pending_chars.pop() {
                        ctx.terminal
                            .set_character_visibility(next_character, true);
                        ctx.active_characters.insert(next_character);
                    }
                }
            }
            ctx.update(self);
            return Some(ctx.frame());
        }
        None
    }
}
