//! expand, ported from effects/effect_expand.py.

use rustc_hash::FxHashMap;

use crate::{
    effects::common::{parse_color, parse_easing, parse_gradient_direction},
    engine::{
        animation::{ExistingColorHandling, SyncMetric, VisualParams},
        character::CharId,
        ctx::{EffectHooks, EngineCtx},
        effect::Effect,
        error::EngineError,
        events::{CallerKey, Event, EventAction},
        terminal::{CharacterFilter, CharacterSort},
    },
    utils::{
        easing::Easing,
        graphics::{Color, ColorPair, Gradient, GradientDirection},
    },
};

#[derive(Debug, Clone)]
pub struct ExpandConfig {
    /// Easing function to use for character movement.
    pub expand_easing: Easing,

    /// Movement speed of the characters.
    pub movement_speed: f64,

    /// Space separated, unquoted, list of colors for the final color gradient.
    pub final_gradient_stops: Vec<Color>,

    /// Number of gradient steps to use.
    pub final_gradient_steps: Vec<i64>,

    /// Direction of the final gradient.
    pub final_gradient_direction: GradientDirection,
}

impl Default for ExpandConfig {
    fn default() -> Self {
        Self {
            expand_easing: parse_easing("in_out_quart").expect("valid literal"),
            movement_speed: 0.35,
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
pub struct Expand {
    config: ExpandConfig,
    character_final_color_map: FxHashMap<CharId, ColorPair>,
}

impl Expand {
    pub fn new(config: ExpandConfig) -> Self {
        Expand {
            config,
            character_final_color_map: FxHashMap::default(),
        }
    }
}

impl EffectHooks for Expand {}
impl Effect for Expand {
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
        for id in characters {
            let (input_fg, input_bg, input_coord) = {
                let ch = &ctx.terminal.arena[id.0 as usize];
                (
                    ch.animation.input_fg_color,
                    ch.animation.input_bg_color,
                    ch.input_coord,
                )
            };
            let final_colors = if dynamic {
                ColorPair::new(input_fg, input_bg)
            } else {
                ColorPair::new(
                    Some(*final_gradient_mapping.get(&input_coord).unwrap()),
                    None,
                )
            };
            self.character_final_color_map.insert(id, final_colors);
        }
        let characters = {
            let filter = CharacterFilter::default();
            ctx.terminal.get_characters(
                &mut ctx.rng,
                filter,
                CharacterSort::TopToBottomLeftToRight,
            )
        };
        for id in characters {
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
            let center = ctx.terminal.canvas.center;
            let input_coord_path = {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.motion.set_coordinate(center);
                let path_id = ch
                    .motion
                    .new_path(
                        self.config.movement_speed,
                        Some(self.config.expand_easing),
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
            ctx.terminal.set_character_visibility(id, true);
            ctx.active_characters.insert(id);
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
            ctx.activate_path(self, id, &input_coord_path);
            let gradient_scn = {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.animation.new_scene(
                    false,
                    Some(SyncMetric::Distance),
                    None,
                    "",
                    uses_pre,
                )
            };
            let symbols = vec![input_symbol.clone()];
            {
                let scene = {
                    let ch = &mut ctx.terminal.arena[id.0 as usize];
                    ch.animation.scenes.get_mut(&gradient_scn).unwrap()
                };
                if dynamic {
                    let fg_gradient = match &input_fg {
                        Some(c) => Some(
                            Gradient::with_steps(
                                &[final_gradient.spectrum[0], *c],
                                10,
                                false,
                            )
                            .map_err(EngineError::Other)?,
                        ),
                        None => None,
                    };
                    let bg_gradient = match &input_bg {
                        Some(c) => Some(
                            Gradient::with_steps(
                                &[final_gradient.spectrum[0], *c],
                                10,
                                false,
                            )
                            .map_err(EngineError::Other)?,
                        ),
                        None => None,
                    };
                    if fg_gradient.is_some() || bg_gradient.is_some() {
                        scene
                            .apply_gradient_to_symbols(
                                &symbols,
                                1,
                                fg_gradient.as_ref(),
                                bg_gradient.as_ref(),
                            )
                            .map_err(EngineError::Other)?;
                    } else {
                        scene
                            .add_frame(
                                &input_symbol,
                                1,
                                VisualParams {
                                    colors: Some(ColorPair::default()),
                                    ..Default::default()
                                },
                            )
                            .map_err(EngineError::Other)?;
                    }
                } else {
                    let final_fg = self.character_final_color_map[&id]
                        .fg_color
                        .expect("gradient mapping fg");
                    let gradient = Gradient::with_steps(
                        &[final_gradient.spectrum[0], final_fg],
                        10,
                        false,
                    )
                    .map_err(EngineError::Other)?;
                    scene
                        .apply_gradient_to_symbols(
                            &symbols,
                            5,
                            Some(&gradient),
                            None,
                        )
                        .map_err(EngineError::Other)?;
                }
            }
            ctx.activate_scene(self, id, &gradient_scn);
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
