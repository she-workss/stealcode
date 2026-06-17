use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum PluginEntry {
    Name(String),
    NameWithOptions(String, FxHashMap<String, Value>),
}
