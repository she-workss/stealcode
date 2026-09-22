//! Static effect registry (replaces upstream pkgutil discovery).
//! GENERATED structure - keep alphabetical by name when adding effects.

pub mod beams;
pub mod binarypath;
pub mod blackhole;
pub mod bouncyballs;
pub mod bubbles;
pub mod burn;
pub mod colorshift;
pub mod common;
pub mod crumble;
pub mod decrypt;
pub mod errorcorrect;
pub mod expand;
pub mod fireworks;
pub mod highlight;
pub mod laseretch;
pub mod matrix;
pub mod middleout;
pub mod orbittingvolley;
pub mod pour;
pub mod print_effect;
pub mod rain;
pub mod random_sequence;
pub mod scattered;
pub mod slide;
pub mod smoke;
pub mod spotlights;
pub mod spray;
pub mod swarm;
pub mod sweep;
pub mod synthgrid;
pub mod thunderstorm;
pub mod unstable;
pub mod waves;
pub mod wipe;

use crate::engine::effect::Effect;

/// (effect name, factory building it from its default config).
const REGISTRY: &[(&str, fn() -> Box<dyn Effect>)] = &[
    ("beams", || {
        Box::new(beams::Beams::new(beams::BeamsConfig::default()))
    }),
    ("binarypath", || {
        Box::new(binarypath::BinaryPath::new(
            binarypath::BinaryPathConfig::default(),
        ))
    }),
    ("blackhole", || {
        Box::new(blackhole::Blackhole::new(
            blackhole::BlackholeConfig::default(),
        ))
    }),
    ("bouncyballs", || {
        Box::new(bouncyballs::BouncyBalls::new(
            bouncyballs::BouncyBallsConfig::default(),
        ))
    }),
    ("bubbles", || {
        Box::new(bubbles::Bubbles::new(bubbles::BubblesConfig::default()))
    }),
    ("burn", || {
        Box::new(burn::Burn::new(burn::BurnConfig::default()))
    }),
    ("colorshift", || {
        Box::new(colorshift::ColorShift::new(
            colorshift::ColorShiftConfig::default(),
        ))
    }),
    ("crumble", || {
        Box::new(crumble::Crumble::new(crumble::CrumbleConfig::default()))
    }),
    ("decrypt", || {
        Box::new(decrypt::Decrypt::new(decrypt::DecryptConfig::default()))
    }),
    ("errorcorrect", || {
        Box::new(errorcorrect::ErrorCorrect::new(
            errorcorrect::ErrorCorrectConfig::default(),
        ))
    }),
    ("expand", || {
        Box::new(expand::Expand::new(expand::ExpandConfig::default()))
    }),
    ("fireworks", || {
        Box::new(fireworks::Fireworks::new(
            fireworks::FireworksConfig::default(),
        ))
    }),
    ("highlight", || {
        Box::new(highlight::Highlight::new(
            highlight::HighlightConfig::default(),
        ))
    }),
    ("laseretch", || {
        Box::new(laseretch::LaserEtch::new(
            laseretch::LaserEtchConfig::default(),
        ))
    }),
    ("matrix", || {
        Box::new(matrix::Matrix::new(matrix::MatrixConfig::default()))
    }),
    ("middleout", || {
        Box::new(middleout::Middleout::new(
            middleout::MiddleoutConfig::default(),
        ))
    }),
    ("orbittingvolley", || {
        Box::new(orbittingvolley::OrbittingVolley::new(
            orbittingvolley::OrbittingVolleyConfig::default(),
        ))
    }),
    ("pour", || {
        Box::new(pour::Pour::new(pour::PourConfig::default()))
    }),
    ("print", || {
        Box::new(print_effect::Print::new(
            print_effect::PrintConfig::default(),
        ))
    }),
    ("rain", || {
        Box::new(rain::Rain::new(rain::RainConfig::default()))
    }),
    ("randomsequence", || {
        Box::new(random_sequence::RandomSequence::new(
            random_sequence::RandomSequenceConfig::default(),
        ))
    }),
    ("scattered", || {
        Box::new(scattered::Scattered::new(
            scattered::ScatteredConfig::default(),
        ))
    }),
    ("slide", || {
        Box::new(slide::Slide::new(slide::SlideConfig::default()))
    }),
    ("smoke", || {
        Box::new(smoke::Smoke::new(smoke::SmokeConfig::default()))
    }),
    ("spotlights", || {
        Box::new(spotlights::Spotlights::new(
            spotlights::SpotlightsConfig::default(),
        ))
    }),
    ("spray", || {
        Box::new(spray::Spray::new(spray::SprayConfig::default()))
    }),
    ("swarm", || {
        Box::new(swarm::Swarm::new(swarm::SwarmConfig::default()))
    }),
    ("sweep", || {
        Box::new(sweep::Sweep::new(sweep::SweepConfig::default()))
    }),
    ("synthgrid", || {
        Box::new(synthgrid::SynthGrid::new(
            synthgrid::SynthGridConfig::default(),
        ))
    }),
    ("thunderstorm", || {
        Box::new(thunderstorm::Thunderstorm::new(
            thunderstorm::ThunderstormConfig::default(),
        ))
    }),
    ("unstable", || {
        Box::new(unstable::Unstable::new(unstable::UnstableConfig::default()))
    }),
    ("waves", || {
        Box::new(waves::Waves::new(waves::WavesConfig::default()))
    }),
    ("wipe", || {
        Box::new(wipe::Wipe::new(wipe::WipeConfig::default()))
    }),
];

/// Names of every registered effect (e.g. "beams", "wipe").
pub fn effect_names() -> Vec<String> {
    REGISTRY
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect()
}

/// Build an effect from its name (e.g. "beams") with pure default config,
/// for library consumers (examples, tests, --random-effect).
pub fn build_effect(name: &str) -> Option<Box<dyn Effect>> {
    let (_, make) =
        REGISTRY.iter().find(|(candidate, _)| *candidate == name)?;
    Some(make())
}

#[cfg(test)]
mod tests {
    use super::{build_effect, effect_names};

    #[test]
    fn every_registered_effect_builds_with_default_config() {
        let names = effect_names();
        let mut sorted: Vec<&str> = names.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        assert!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "effect names must be unique"
        );
        assert!(names.len() >= 30, "expected the full effect registry");
        for name in &names {
            assert!(
                build_effect(name).is_some(),
                "failed to build effect '{name}'"
            );
        }
    }
}
