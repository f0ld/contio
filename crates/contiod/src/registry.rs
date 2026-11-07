//! Daemon state that survives restarts besides jobs (spec §6): models
//! registered through the API and user settings. Both are JSON files in the
//! data directory; `config.toml` stays the user's hand-edited file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use std::collections::BTreeMap;

use contio_backends::{BackendError, ModelManager};
use contio_core::hw::GpuApi;
use contio_core::model::ModelProfile;
use contio_install::Method;
use serde::{Deserialize, Serialize};

use crate::catalog_api::ModelPatch;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// How missing backends are installed when a model needs one.
    pub llamacpp_method: Method,
    pub flm_method: Method,
    pub whisper_method: Method,
    /// GPU API for GGUF models on the GPU that don't pick one (m0-gpu.md §8: Vulkan is faster overall).
    pub gpu_api: GpuApi,
    /// Hold the `performance` power profile while a model works (docs/PERFORMANCE.md §2e).
    pub serve_power: crate::boost::ServePower,
}

impl Default for Settings {
    fn default() -> Self {
        // prebuilt: no build dependencies, ~30 MB, verified in M1
        Self {
            llamacpp_method: Method::Prebuilt,
            flm_method: Method::Source,
            whisper_method: Method::Source,
            gpu_api: GpuApi::Vulkan,
            serve_power: crate::boost::ServePower::Off,
        }
    }
}

/// Saves managed model profiles, overrides of `config.toml` models, and settings.
#[derive(Clone)]
pub struct Registry {
    models_path: PathBuf,
    settings_path: PathBuf,
    settings: Arc<RwLock<Settings>>,
    overrides_path: PathBuf,
    /// Changed launch settings of `config.toml` models, by id.
    overrides: Arc<RwLock<BTreeMap<String, ModelPatch>>>,
    /// The `config.toml` models as written, for resetting overrides.
    config_models: Arc<Vec<ModelProfile>>,
    /// One writer per file at a time.
    saving: Arc<Mutex<()>>,
}

impl Registry {
    pub fn new(data_dir: &Path) -> Self {
        let settings_path = data_dir.join("settings.json");
        let settings = read_json(&settings_path).unwrap_or_default();
        let overrides_path = data_dir.join("overrides.json");
        Self {
            models_path: data_dir.join("models.json"),
            settings_path,
            settings: Arc::new(RwLock::new(settings)),
            overrides: Arc::new(RwLock::new(read_json(&overrides_path).unwrap_or_default())),
            overrides_path,
            config_models: Arc::default(),
            saving: Default::default(),
        }
    }

    /// Remember the `config.toml` models and apply their saved overrides to
    /// `models` (before the manager is built). Overrides that no longer apply
    /// (model gone, invalid value) are logged and skipped.
    pub fn apply_overrides(&mut self, models: &mut [ModelProfile]) {
        self.config_models = Arc::new(models.to_vec());
        let overrides = self.overrides.read().expect("overrides lock poisoned");
        for (id, patch) in overrides.iter() {
            match models.iter_mut().find(|m| &m.id == id) {
                Some(m) => {
                    if let Err(e) = patch.apply(m) {
                        tracing::warn!(model = %id, "ignoring saved override: {e}");
                    }
                }
                None => tracing::warn!(model = %id, "saved override for a model no longer in config.toml"),
            }
        }
    }

    pub fn config_profile(&self, id: &str) -> Option<ModelProfile> {
        self.config_models.iter().find(|m| m.id == id).cloned()
    }

    pub fn has_override(&self, id: &str) -> bool {
        self.overrides.read().expect("overrides lock poisoned").get(id).is_some_and(|p| *p != ModelPatch::default())
    }

    /// Fold `patch` into the saved overrides of a `config.toml` model.
    pub fn add_override(&self, id: &str, patch: &ModelPatch) -> std::io::Result<()> {
        let _saving = self.saving.lock().expect("registry mutex poisoned");
        let mut all = self.overrides.write().expect("overrides lock poisoned");
        all.entry(id.to_owned()).or_default().merge(patch);
        write_json(&self.overrides_path, &*all)
    }

    pub fn clear_override(&self, id: &str) -> std::io::Result<()> {
        let _saving = self.saving.lock().expect("registry mutex poisoned");
        let mut all = self.overrides.write().expect("overrides lock poisoned");
        all.remove(id);
        write_json(&self.overrides_path, &*all)
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

#[cfg(test)]
mod tests {
    use contio_core::hw::DeviceKind;
    use contio_core::model::ModelSource;

    use super::*;

    #[test]
    fn config_overrides_persist_and_reset() {
        let dir = tempfile::tempdir().unwrap();
        let model = || ModelProfile::new("coder", ModelSource::Gguf { path: "/m/c.gguf".into(), mmproj: None }, DeviceKind::Gpu);
        let mut original = model();
        original.context = Some(32768);

        let mut reg = Registry::new(dir.path());
        let mut models = vec![original.clone()];
        reg.apply_overrides(&mut models);
        assert_eq!(models[0], original, "nothing saved yet");
        assert!(!reg.has_override("coder"));

        reg.add_override("coder", &ModelPatch { context: Some(131072), ..Default::default() }).unwrap();
        reg.add_override("coder", &ModelPatch { pinned: Some(true), ..Default::default() }).unwrap();
        assert!(reg.has_override("coder"));

        // next start: both changes applied on top of config.toml
        let mut reg = Registry::new(dir.path());
        let mut models = vec![original.clone()];
        reg.apply_overrides(&mut models);
        assert_eq!((models[0].context, models[0].pinned), (Some(131072), true));
        assert_eq!(reg.config_profile("coder").unwrap(), original, "the file's version is kept for resets");

        reg.clear_override("coder").unwrap();
        let mut reg = Registry::new(dir.path());
        let mut models = vec![original.clone()];
        reg.apply_overrides(&mut models);
        assert_eq!(models[0], original);
    }
}
