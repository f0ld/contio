//! Daemon state that survives restarts besides jobs (spec §6): models
//! registered through the API and user settings. Both are JSON files in the
//! data directory; `config.toml` stays the user's hand-edited file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use contio_backends::{BackendError, ModelManager};
use contio_core::model::ModelProfile;
use contio_install::Method;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// How missing backends are installed when a model needs one.
    pub llamacpp_method: Method,
    pub flm_method: Method,
    pub whisper_method: Method,
}

impl Default for Settings {
    fn default() -> Self {
        // prebuilt: no build dependencies, ~30 MB, verified in M1
        Self { llamacpp_method: Method::Prebuilt, flm_method: Method::Source, whisper_method: Method::Source }
    }
}

/// Saves managed model profiles and settings.
#[derive(Clone)]
pub struct Registry {
    models_path: PathBuf,
    settings_path: PathBuf,
    settings: Arc<RwLock<Settings>>,
    /// One writer per file at a time.
    saving: Arc<Mutex<()>>,
}

impl Registry {
    pub fn new(data_dir: &Path) -> Self {
        let settings_path = data_dir.join("settings.json");
        let settings = read_json(&settings_path).unwrap_or_default();
        Self {
            models_path: data_dir.join("models.json"),
            settings_path,
            settings: Arc::new(RwLock::new(settings)),
            saving: Default::default(),
        }
    }

    /// Register the saved models with `manager`. A saved model whose id is now
    /// taken by `config.toml` is skipped (config wins) and kept in the file.
    pub fn restore(&self, manager: &ModelManager) {
        let saved: Vec<ModelProfile> = read_json(&self.models_path).unwrap_or_default();
        for p in saved {
            let id = p.id.clone();
            match manager.add_profile(p, true) {
                Ok(()) => {}
                Err(BackendError::DuplicateModel(_)) => {
                    tracing::warn!(model = %id, "saved model hidden by a config.toml model with the same id");
                }
                Err(e) => tracing::warn!(model = %id, "not restoring saved model: {e}"),
            }
        }
    }

    /// Persist the manager's managed profiles.
    pub fn save_models(&self, manager: &ModelManager) {
        let _saving = self.saving.lock().expect("registry mutex poisoned");
        let mut profiles = manager.managed_profiles();
        // keep entries hidden by a config.toml model of the same id (see `restore`)
        let config_ids: Vec<String> =
            manager.profiles().into_iter().map(|p| p.id).filter(|id| !profiles.iter().any(|p| &p.id == id)).collect();
        let saved: Vec<ModelProfile> = read_json(&self.models_path).unwrap_or_default();
        profiles.extend(saved.into_iter().filter(|s| config_ids.contains(&s.id)));
        if let Err(e) = write_json(&self.models_path, &profiles) {
            tracing::error!("saving {}: {e}", self.models_path.display());
        }
    }

    pub fn settings(&self) -> Settings {
        self.settings.read().expect("settings lock poisoned").clone()
    }

    pub fn set_settings(&self, settings: Settings) -> std::io::Result<()> {
        let _saving = self.saving.lock().expect("registry mutex poisoned");
        write_json(&self.settings_path, &settings)?;
        *self.settings.write().expect("settings lock poisoned") = settings;
        Ok(())
    }
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes)
        .map_err(|e| tracing::warn!("ignoring unreadable {}: {e}", path.display()))
        .ok()
}

pub fn write_json(path: &Path, value: &impl Serialize) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(value).map_err(std::io::Error::other)?)?;
    std::fs::rename(&tmp, path)
}
