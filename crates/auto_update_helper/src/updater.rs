//! Job-based, rollback-capable file swap - the same shape Zed uses in its
//! `auto_update_helper::updater`, written independently for StealCode.
//!
//! No Restart Manager here: Zed needs it because
//! `explorer_command_injector.dll` is a COM shell extension that Explorer keeps
//! loaded. StealCode's context menu is plain registry `command` strings -
//! nothing but our own process ever has `stealcode.exe` open, so a short retry
//! loop is enough to ride out the last moments of our own shutdown.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};

pub struct Job {
    pub apply: Box<dyn Fn(&Path) -> Result<()> + Send + Sync>,
    pub rollback: Box<dyn Fn(&Path) -> Result<()> + Send + Sync>,
}

impl Job {
    #[must_use]
    pub fn mkdir(name: &'static str) -> Self {
        Job {
            apply: Box::new(move |app_dir| {
                std::fs::create_dir_all(app_dir.join(name))
                    .with_context(|| format!("failed to create {name}"))
            }),
            rollback: Box::new(move |app_dir| {
                std::fs::remove_dir_all(app_dir.join(name))
                    .with_context(|| format!("failed to remove {name}"))
            }),
        }
    }

    #[must_use]
    pub fn move_file(from: &'static str, to: &'static str) -> Self {
        Job {
            apply: Box::new(move |app_dir| {
                std::fs::rename(app_dir.join(from), app_dir.join(to))
                    .with_context(|| format!("failed to move {from} -> {to}"))
            }),
            rollback: Box::new(move |app_dir| {
                std::fs::rename(app_dir.join(to), app_dir.join(from))
                    .with_context(|| {
                        format!("failed to roll back {from} -> {to}")
                    })
            }),
        }
    }

    #[must_use]
    pub fn rmdir_nofail(name: &'static str) -> Self {
        Job {
            apply: Box::new(move |app_dir| {
                if let Err(error) = std::fs::remove_dir_all(app_dir.join(name))
                {
                    tracing::warn!("failed to remove {name}: {error}");
                }
                Ok(())
            }),
            rollback: Box::new(move |_| {
                anyhow::bail!("delete of {name} cannot be rolled back")
            }),
        }
    }
}

/// StealCode only manages one file, so this list is much shorter than
/// Zed's (which also juggles `bin\zed.exe`, `conpty.dll`, `OpenConsole.exe`
/// for x64/arm64). Extend this if StealCode ever ships more than one
/// binary into the install directory.
#[must_use]
pub fn jobs() -> Vec<Job> {
    vec![
        Job::mkdir("old"),
        Job::move_file("stealcode.exe", "old\\stealcode.exe"),
        Job::move_file("install\\stealcode.exe", "stealcode.exe"),
        Job::rmdir_nofail("updates"),
        Job::rmdir_nofail("install"),
        Job::rmdir_nofail("old"),
    ]
}

/// Applies every job in order with a short retry loop per job (handles the
/// brief window where our own parent process might still be finishing
/// shutdown). Rolls back everything already applied if a job never
/// succeeds within `per_job_timeout`.
pub fn perform_update_with_timeout(
    app_dir: &Path,
    launch: bool,
    per_job_timeout: Duration,
) -> Result<()> {
    let jobs = jobs();
    let mut last_successful: Option<usize> = None;

    'outer: for (index, job) in jobs.iter().enumerate() {
        let start = Instant::now();
        loop {
            match (job.apply)(app_dir) {
                Ok(()) => {
                    last_successful = Some(index);
                    break;
                }
                Err(error) => {
                    if start.elapsed() > per_job_timeout {
                        tracing::error!(
                            "job {index} timed out ({error}), rolling back"
                        );
                        break 'outer;
                    }
                    tracing::warn!("job {index} failed (retrying): {error}");
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }

    if last_successful != Some(jobs.len() - 1) {
        if let Some(last) = last_successful {
            for job in jobs[..=last].iter().rev() {
                if let Err(error) = (job.rollback)(app_dir) {
                    anyhow::bail!(
                        "rollback failed, app may be inconsistent: {error}"
                    );
                }
            }
        }
        anyhow::bail!("update failed, rolled back");
    }

    if launch {
        let _ =
            std::process::Command::new(app_dir.join("stealcode.exe")).spawn();
    }
    Ok(())
}

/// Real entry point used by `main.rs` - a 2 second per-job timeout, which
/// is generous for a rename/mkdir/rmdir on local disk even under load.
pub fn perform_update(app_dir: &Path, launch: bool) -> Result<()> {
    perform_update_with_timeout(app_dir, launch, Duration::from_secs(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAST_TIMEOUT: Duration = Duration::from_millis(100);

    #[test]
    fn perform_update_applies_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let app_dir = dir.path();
        std::fs::write(app_dir.join("stealcode.exe"), b"old-version").unwrap();
        std::fs::create_dir_all(app_dir.join("install")).unwrap();
        std::fs::write(app_dir.join("install/stealcode.exe"), b"new-version")
            .unwrap();
        std::fs::create_dir_all(app_dir.join("updates")).unwrap();

        perform_update_with_timeout(app_dir, false, FAST_TIMEOUT)
            .expect("update should succeed");

        assert_eq!(
            std::fs::read(app_dir.join("stealcode.exe")).unwrap(),
            b"new-version"
        );
        assert!(!app_dir.join("install").exists());
        assert!(!app_dir.join("updates").exists());
        assert!(!app_dir.join("old").exists());
    }

    #[test]
    fn perform_update_rolls_back_when_install_source_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let app_dir = dir.path();
        std::fs::write(app_dir.join("stealcode.exe"), b"old-version").unwrap();
        // Deliberately no install\stealcode.exe: job 3 (move
        // install\stealcode.exe -> stealcode.exe) will fail every retry
        // and time out.

        let result = perform_update_with_timeout(app_dir, false, FAST_TIMEOUT);

        assert!(result.is_err());
        // Job 1 (backing up stealcode.exe to old\) must have been rolled back.
        assert!(app_dir.join("stealcode.exe").exists());
        assert_eq!(
            std::fs::read(app_dir.join("stealcode.exe")).unwrap(),
            b"old-version"
        );
        assert!(!app_dir.join("old").join("stealcode.exe").exists());
    }

    #[test]
    fn perform_update_does_not_launch_when_launch_is_false() {
        let dir = tempfile::tempdir().unwrap();
        let app_dir = dir.path();
        std::fs::write(app_dir.join("stealcode.exe"), b"old").unwrap();
        std::fs::create_dir_all(app_dir.join("install")).unwrap();
        std::fs::write(app_dir.join("install/stealcode.exe"), b"new").unwrap();
        std::fs::create_dir_all(app_dir.join("updates")).unwrap();

        // `stealcode.exe` after the swap is just a text file, not a real
        // executable, so if this incorrectly tried to launch it, `spawn()`
        // would either fail silently (ignored via `let _ =`) or launch
        // something nonsensical - this test mainly documents the contract,
        // not the failure mode.
        perform_update_with_timeout(app_dir, false, FAST_TIMEOUT).unwrap();
    }
}
