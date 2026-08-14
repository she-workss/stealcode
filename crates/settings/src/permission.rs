use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionAction {
    Ask,
    Allow,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PermissionRule {
    Action(PermissionAction),
    PerTool(FxHashMap<String, PermissionAction>),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PermissionDetails {
    pub read: Option<PermissionRule>,
    pub edit: Option<PermissionRule>,
    pub glob: Option<PermissionRule>,
    pub grep: Option<PermissionRule>,
    pub list: Option<PermissionRule>,
    pub bash: Option<PermissionRule>,
    pub task: Option<PermissionRule>,
    pub external_directory: Option<PermissionRule>,
    pub todowrite: Option<PermissionAction>,
    pub question: Option<PermissionAction>,
    pub webfetch: Option<PermissionAction>,
    pub websearch: Option<PermissionAction>,
    pub lsp: Option<PermissionRule>,
    pub doom_loop: Option<PermissionAction>,
    pub skill: Option<PermissionRule>,
    #[serde(flatten)]
    pub extra: FxHashMap<String, PermissionRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PermissionSettings {
    Action(PermissionAction),
    Detailed(Box<PermissionDetails>),
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn permission_config_action_string() {
        let v: PermissionSettings =
            serde_json::from_value(json!("allow")).unwrap();
        assert!(matches!(
            v,
            PermissionSettings::Action(PermissionAction::Allow)
        ));
    }

    #[test]
    fn permission_config_detailed_object() {
        let v: PermissionSettings = serde_json::from_value(json!({
            "read": "allow",
            "edit": "ask",
            "bash": { "ls": "allow", "rm": "deny" },
            "todowrite": "deny"
        }))
        .unwrap();
        let PermissionSettings::Detailed(details) = v else {
            panic!("expected Detailed variant");
        };
        assert_eq!(
            details.read,
            Some(PermissionRule::Action(PermissionAction::Allow))
        );
        assert_eq!(details.todowrite, Some(PermissionAction::Deny));
        let PermissionRule::PerTool(bash_map) = details.bash.unwrap() else {
            panic!("expected PerTool");
        };
        assert_eq!(bash_map["rm"], PermissionAction::Deny);
    }

    #[test]
    fn permission_config_unknown_tool_in_extra() {
        let v: PermissionSettings = serde_json::from_value(json!({
            "my_custom_tool": "allow"
        }))
        .unwrap();
        let PermissionSettings::Detailed(details) = v else {
            panic!("expected Detailed variant");
        };
        assert_eq!(
            details.extra["my_custom_tool"],
            PermissionRule::Action(PermissionAction::Allow)
        );
    }
}
