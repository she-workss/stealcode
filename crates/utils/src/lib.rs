use std::path::{Path, PathBuf};

pub mod paths;
pub mod rel_path;

/// Normalizes a path by resolving `.` and `..` components without
/// requiring the path to exist on disk (unlike `canonicalize`).
#[must_use]
pub fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut components = path.components().peekable();
    let mut ret =
        if let Some(c @ Component::Prefix(..)) = components.peek().copied() {
            components.next();
            PathBuf::from(c.as_os_str())
        } else {
            PathBuf::new()
        };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::Normal(c) => {
                ret.push(c);
            }
        }
    }
    ret
}
