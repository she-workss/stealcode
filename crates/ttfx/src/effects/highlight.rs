//! highlight, ported from effects/effect_highlight.py.

use crate::{
    effects::common::{
        parse_character_group, parse_color, parse_gradient_direction,
    },
    engine::{
        animation::{Animation, ExistingColorHandling, VisualParams},
        character::CharId,
        ctx::{EffectHooks, EngineCtx},
        effect::Effect,
        error::EngineError,
        terminal::{CharacterFilter, CharacterGroup, CharacterSort},
    },
    utils::{
        easing::{Easing, SequenceEaser},
        graphics::{Color, ColorPair, Gradient, GradientDirection},
    },
};

#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Brightness of the highlight color.
    pub highlight_brightness: f64,

    /// Direction the highlight will travel.
    pub highlight_direction: CharacterGroup,

    /// Width of the highlight. n >= 1
    pub highlight_width: i64,

    /// Space separated, unquoted, list of colors for the final color gradient.
    pub final_gradient_stops: Vec<Color>,

    /// Number of gradient steps to use.
    pub final_gradient_steps: Vec<i64>,

    /// Direction of the final gradient.
    pub final_gradient_direction: GradientDirection,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            highlight_brightness: 1.75,
            highlight_direction: parse_character_group(
                "diagonal_bottom_left_to_top_right",
            )
            .expect("valid literal"),
            highlight_width: 8,
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
pub struct Highlight {
    config: HighlightConfig,
    easer: Option<SequenceEaser<Vec<CharId>>>,
}

impl Highlight {
    pub fn new(config: HighlightConfig) -> Self {
        Highlight {
            config,
            easer: None,
        }
    }
}

impl EffectHooks for Highlight {}
impl Effect for Highlight {
    fn build(&mut self, ctx: &mut EngineCtx) -> Result<(), EngineError> {
        let groups = ctx.terminal.get_characters_grouped(
            CharacterFilter::default(),
            self.config.highlight_direction,
        );
        self.easer = Some(SequenceEaser::new(groups, Easing::InOutCirc, 100));

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
            let (base_color, input_bg_color) = if dynamic {
                (input_fg, input_bg)
            } else {
                (
                    Some(*final_gradient_mapping.get(&input_coord).unwrap()),
                    None,
                )
            };
            let base_colors = ColorPair::new(base_color, input_bg_color);
            let highlight_gradient = match &base_color {
                Some(base) => {
                    let highlight_color = Animation::adjust_color_brightness(
                        base,
                        self.config.highlight_brightness,
                    );
                    Some(
                        Gradient::new(
                            &[*base, highlight_color, highlight_color, *base],
                            &[3, self.config.highlight_width, 3],
                            false,
                            false,
                        )
                        .map_err(EngineError::Other)?,
                    )
                }
                None => None,
            };
            {
                let ch = &mut ctx.terminal.arena[id.0 as usize];
                ch.animation.set_appearance(
                    &input_symbol,
                    uses_pre,
                    Some(&input_symbol),
                    Some(base_colors),
                );
                ch.animation.new_scene(
                    false,
                    None,
                    None,
                    "highlight",
                    uses_pre,
                );
                let scene = ch.animation.scenes.get_mut("highlight").unwrap();
                if let Some(gradient) = &highlight_gradient {
                    for color in &gradient.spectrum {
                        scene
                            .add_frame(
                                &input_symbol,
                                2,
                                VisualParams {
                                    colors: Some(ColorPair::new(
                                        Some(*color),
                                        input_bg_color,
                                    )),
                                    ..Default::default()
                                },
                            )
                            .map_err(EngineError::Other)?;
                    }
                } else {
                    scene
                        .add_frame(
                            &input_symbol,
                            2,
                            VisualParams {
                                colors: Some(base_colors),
                                ..Default::default()
                            },
                        )
                        .map_err(EngineError::Other)?;
                }
            }
            ctx.terminal.set_character_visibility(id, true);
        }
        Ok(())
    }

    fn next_frame(&mut self, ctx: &mut EngineCtx) -> Option<String> {
        let easer_complete = self.easer.as_ref().unwrap().is_complete();
        if !ctx.active_characters.is_empty() || !easer_complete {
            let mut easer = self.easer.take().unwrap();
            let step = easer.step();
            for group in step.added {
                for &id in group {
                    ctx.activate_scene(self, id, "highlight");
                    ctx.active_characters.insert(id);
                }
            }
            self.easer = Some(easer);
            ctx.update(self);
            return Some(ctx.frame());
        }
        None
    }
}
