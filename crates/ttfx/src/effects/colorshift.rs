//! colorshift, ported from effects/effect_colorshift.py.

use rustc_hash::FxHashMap;

use crate::{
    effects::common::{parse_color, parse_gradient_direction},
    engine::{
        animation::{ExistingColorHandling, VisualParams},
        character::CharId,
        ctx::{EffectHooks, EngineCtx},
        effect::Effect,
        error::EngineError,
        events::{CallerKey, EffectCallback, Event, EventAction},
        terminal::{CharacterFilter, CharacterSort},
    },
    utils::{
        geometry,
        graphics::{Color, ColorPair, Gradient, GradientDirection},
    },
};

#[derive(Debug, Clone)]
pub struct ColorShiftConfig {
    /// Space separated, unquoted, list of colors for the gradient.
    pub gradient_stops: Vec<Color>,

    /// Number of gradient steps to use. More steps will create a smoother
    /// gradient animation.
    pub gradient_steps: Vec<i64>,

    /// Number of frames to display each gradient step. Increase to slow down
    /// the gradient animation.
    pub gradient_frames: i64,

    /// Do not display the gradient as a wave.
    pub no_travel: bool,

    /// Direction the gradient travels across the canvas.
    pub travel_direction: GradientDirection,

    /// Reverse the gradient travel direction.
    pub reverse_travel_direction: bool,

    /// Do not loop the gradient.
    pub no_loop: bool,

    /// Number of times to cycle the gradient.
    pub cycles: i64,

    /// Skip the final gradient.
    pub skip_final_gradient: bool,

    /// Space separated, unquoted, list of colors for the final color gradient.
    pub final_gradient_stops: Vec<Color>,

    /// Number of gradient steps to use for the final gradient.
    pub final_gradient_steps: Vec<i64>,

    /// Direction of the final gradient.
    pub final_gradient_direction: GradientDirection,
}

impl Default for ColorShiftConfig {
    fn default() -> Self {
        Self {
            gradient_stops: vec![
                parse_color("e81416").expect("valid color"),
                parse_color("ffa500").expect("valid color"),
                parse_color("faeb36").expect("valid color"),
                parse_color("79c314").expect("valid color"),
                parse_color("487de7").expect("valid color"),
                parse_color("4b369d").expect("valid color"),
                parse_color("70369d").expect("valid color"),
            ],
            gradient_steps: vec![12],
            gradient_frames: 2,
            no_travel: false,
            travel_direction: parse_gradient_direction("radial")
                .expect("valid literal"),
            reverse_travel_direction: false,
            no_loop: false,
            cycles: 3,
            skip_final_gradient: false,
            final_gradient_stops: vec![
                parse_color("e81416").expect("valid color"),
                parse_color("ffa500").expect("valid color"),
                parse_color("faeb36").expect("valid color"),
                parse_color("79c314").expect("valid color"),
                parse_color("487de7").expect("valid color"),
                parse_color("4b369d").expect("valid color"),
                parse_color("70369d").expect("valid color"),
            ],
            final_gradient_steps: vec![12],
            final_gradient_direction: parse_gradient_direction("vertical")
                .expect("valid literal"),
        }
    }
}

#[derive(Debug)]
pub struct ColorShift {
    config: ColorShiftConfig,
    character_final_color_map: FxHashMap<CharId, Color>,
    loop_tracker_map: FxHashMap<CharId, i64>,
}

impl ColorShift {
    pub fn new(config: ColorShiftConfig) -> Self {
        ColorShift {
            config,
            character_final_color_map: FxHashMap::default(),
            loop_tracker_map: FxHashMap::default(),
        }
    }
}

impl EffectHooks for ColorShift {
    /// ColorShiftIterator.loop_tracker.
    fn dispatch_callback(
        &mut self,
        ctx: &mut EngineCtx,
        character: CharId,
        _callback: &EffectCallback,
    ) {
        let count = {
            let entry = self.loop_tracker_map.entry(character).or_insert(0);
            *entry += 1;
            *entry
        };
        if self.config.cycles == 0 || count < self.config.cycles {
            ctx.activate_scene(self, character, "gradient");
        } else if !self.config.skip_final_gradient {
            ctx.activate_scene(self, character, "final_gradient");
        }
    }
}

impl Effect for ColorShift {
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
        let characters = {
            let filter = CharacterFilter::default();
            ctx.terminal.get_characters(
                &mut ctx.rng,
                filter,
                CharacterSort::TopToBottomLeftToRight,
            )
        };
        for &id in &characters {
            let input_coord = ctx.terminal.arena[id.0 as usize].input_coord;
            self.character_final_color_map
                .insert(id, *final_gradient_mapping.get(&input_coord).unwrap());
        }
        let gradient = Gradient::new(
            &self.config.gradient_stops,
            &self.config.gradient_steps,
            false,
            !self.config.no_loop,
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
        for id in characters {
            ctx.terminal.set_character_visibility(id, true);
            let (input_fg, input_bg, input_coord, input_symbol, uses_pre) = {
                let ch = &ctx.terminal.arena[id.0 as usize];
                (
                    ch.animation.input_fg_color,
                    ch.animation.input_bg_color,
                    ch.input_coord,
                    ch.input_symbol.clone(),
                    ch.uses_input_preexisting_colors,
                )
            };
            let colors: Vec<Color> = if self.config.no_travel {
                gradient.spectrum.clone()
            } else {
                let direction_index = match self.config.travel_direction {
                    GradientDirection::Horizontal => {
                        input_coord.column as f64
                            / ctx.terminal.canvas.right as f64
                    }
                    GradientDirection::Vertical => {
                        input_coord.row as f64 / ctx.terminal.canvas.top as f64
                    }
                    GradientDirection::Diagonal => {
                        (input_coord.row + input_coord.column) as f64
                            / (ctx.terminal.canvas.right
                                + ctx.terminal.canvas.top)
                                as f64
                    }
                    GradientDirection::Radial => {
                        geometry::find_normalized_distance_from_center(
                            ctx.terminal.canvas.text_bottom,
                            ctx.terminal.canvas.text_top,
                            ctx.terminal.canvas.text_left,
                            ctx.terminal.canvas.text_right,
                            input_coord,
                        )
                        .map_err(EngineError::Other)?
                    }
                };
                // int() truncation
                let mut shift_distance =
                    (gradient.spectrum.len() as f64 * direction_index) as i64;
                if self.config.reverse_travel_direction {
                    shift_distance *= -1;
                }
                // Python slicing: spectrum[shift:] + spectrum[:shift], negative
                // shifts wrap
                let len = gradient.spectrum.len() as i64;
                let k = if shift_distance < 0 {
                    (len + shift_distance).max(0) as usize
                } else {
                    shift_distance.min(len) as usize
                };
                let mut rotated = gradient.spectrum[k..].to_vec();
                rotated.extend_from_slice(&gradient.spectrum[..k]);
                rotated
            };
            {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.animation
                    .new_scene(false, None, None, "gradient", uses_pre);
                let scene = ch.animation.scenes.get_mut("gradient").unwrap();
                for color in &colors {
                    scene
                        .add_frame(
                            &input_symbol,
                            self.config.gradient_frames,
                            VisualParams {
                                colors: Some(ColorPair::new(
                                    Some(*color),
                                    None,
                                )),
                                ..Default::default()
                            },
                        )
                        .map_err(EngineError::Other)?;
                }
                ch.animation.new_scene(
                    false,
                    None,
                    None,
                    "final_gradient",
                    uses_pre,
                );
            }
            let last_color = *colors.last().unwrap();
            if dynamic {
                let fg_gradient = match &input_fg {
                    Some(c) => Some(
                        Gradient::with_steps(&[last_color, *c], 8, false)
                            .map_err(EngineError::Other)?,
                    ),
                    None => None,
                };
                let bg_gradient = match &input_bg {
                    Some(c) => Some(
                        Gradient::with_steps(&[last_color, *c], 8, false)
                            .map_err(EngineError::Other)?,
                    ),
                    None => None,
                };
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                let scene =
                    ch.animation.scenes.get_mut("final_gradient").unwrap();
                if fg_gradient.is_some() || bg_gradient.is_some() {
                    scene
                        .apply_gradient_to_symbols(
                            std::slice::from_ref(&input_symbol),
                            self.config.gradient_frames,
                            fg_gradient.as_ref(),
                            bg_gradient.as_ref(),
                        )
                        .map_err(EngineError::Other)?;
                } else {
                    scene
                        .add_frame(
                            &input_symbol,
                            self.config.gradient_frames,
                            VisualParams {
                                colors: Some(ColorPair::default()),
                                ..Default::default()
                            },
                        )
                        .map_err(EngineError::Other)?;
                }
            } else {
                let final_color =
                    *self.character_final_color_map.get(&id).unwrap();
                let final_scene_gradient =
                    Gradient::with_steps(&[last_color, final_color], 8, false)
                        .map_err(EngineError::Other)?;
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                let scene =
                    ch.animation.scenes.get_mut("final_gradient").unwrap();
                for color in &final_scene_gradient.spectrum {
                    scene
                        .add_frame(
                            &input_symbol,
                            self.config.gradient_frames,
                            VisualParams {
                                colors: Some(ColorPair::new(
                                    Some(*color),
                                    None,
                                )),
                                ..Default::default()
                            },
                        )
                        .map_err(EngineError::Other)?;
                }
            }
            ctx.activate_scene(self, id, "gradient");
            ctx.active_characters.insert(id);
            ctx.register_event(
                id,
                Event::SceneComplete,
                CallerKey::Scene("gradient".to_string()),
                EventAction::Callback(EffectCallback {
                    id: 0,
                    args: Vec::new(),
                }),
            )
            .map_err(EngineError::Other)?;
        }
        Ok(())
    }

    fn next_frame(&mut self, ctx: &mut EngineCtx) -> Option<String> {
        if !ctx.active_characters.is_empty() {
            ctx.update(self);
            return Some(ctx.frame());
        }
        None
    }
}
