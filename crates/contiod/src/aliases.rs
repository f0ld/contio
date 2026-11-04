//! Model aliases (spec §7.2): other names for a model, e.g. `coder` →
//! `qwen3.6-35b-a3b`. Aliases from `config.toml` are fixed; aliases added
//! through the API are saved in `<data_dir>/aliases.json`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use axum::Json;
use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use contio_backends::ModelManager;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::AppState;
use crate::config::Alias;
use crate::install_api::{bad_request, not_found};
use crate::registry::{read_json, write_json};

#[derive(Debug, Clone, Serialize)]
pub struct AliasView {
    pub name: String,
    pub model: String,
    pub upstream_model: Option<String>,
    /// Added through the API (else from `config.toml`, read-only).
    pub managed: bool,
}

#[derive(Debug, PartialEq)]
pub enum AliasError {
    Invalid(String),
    Conflict(String),
    NotFound(String),
    InConfig(String),
}

#[derive(Clone)]
pub struct Aliases {
    map: Arc<RwLock<BTreeMap<String, AliasView>>>,
    /// `None`: nothing is saved (tests).
    path: Option<PathBuf>,
    saving: Arc<Mutex<()>>,
}

impl Aliases {
    /// Config aliases plus the saved ones. A saved alias that now clashes
    /// with a model id or a config alias, or whose model is gone, is dropped.
    pub fn new(config: Vec<Alias>, path: Option<PathBuf>, manager: &ModelManager) -> Self {
        let mut map: BTreeMap<String, AliasView> = config.into_iter().map(|a| (a.name.clone(), view(a, false))).collect();
        let saved: Vec<Alias> = path.as_deref().and_then(read_json).unwrap_or_default();
        let ids: Vec<String> = manager.profiles().into_iter().map(|p| p.id).collect();
        for a in saved {
            if map.contains_key(&a.name) || ids.contains(&a.name) {
                tracing::warn!(alias = %a.name, "saved alias clashes with a config alias or model id; dropped");
            } else if !ids.contains(&a.model) {
                tracing::warn!(alias = %a.name, model = %a.model, "saved alias points at a missing model; dropped");
            } else {
                map.insert(a.name.clone(), view(a, true));
            }
        }
        Self { map: Arc::new(RwLock::new(map)), path, saving: Default::default() }
    }

    pub fn get(&self, name: &str) -> Option<AliasView> {
        self.map.read().expect("alias lock poisoned").get(name).cloned()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.read().expect("alias lock poisoned").contains_key(name)
    }

    /// Sorted by name.
    pub fn list(&self) -> Vec<AliasView> {
        self.map.read().expect("alias lock poisoned").values().cloned().collect()
    }

    /// Add or change an API alias.
    pub fn set(&self, alias: Alias, manager: &ModelManager) -> Result<AliasView, AliasError> {
        if !crate::catalog_api::valid_id(&alias.name) {
            return Err(AliasError::Invalid(
                "alias names use lowercase letters, digits and `._:-` (max 64 characters)".into(),
            ));
        }
        let ids: Vec<String> = manager.profiles().into_iter().map(|p| p.id).collect();
        if ids.contains(&alias.name) {
            return Err(AliasError::Conflict(format!("`{}` is already a model id", alias.name)));
        }
        if !ids.contains(&alias.model) {
            return Err(AliasError::NotFound(format!("no model `{}`", alias.model)));
        }
        let entry = view(alias, true);
        {
            let mut map = self.map.write().expect("alias lock poisoned");
            if map.get(&entry.name).is_some_and(|a| !a.managed) {
                return Err(AliasError::InConfig(format!("alias `{}` is set in config.toml", entry.name)));
            }
            map.insert(entry.name.clone(), entry.clone());
        }
        self.save();
        Ok(entry)
    }

    pub fn remove(&self, name: &str) -> Result<AliasView, AliasError> {
        let removed = {
            let mut map = self.map.write().expect("alias lock poisoned");
            match map.get(name) {
                None => return Err(AliasError::NotFound(format!("no alias `{name}`"))),
                Some(a) if !a.managed => {
                    return Err(AliasError::InConfig(format!("alias `{name}` is set in config.toml")));
                }
                Some(_) => map.remove(name).expect("checked above"),
            }
        };
        self.save();
        Ok(removed)
    }

    /// Drop the API aliases of a removed model. Returns their names.
    pub fn remove_for_model(&self, model: &str) -> Vec<String> {
        let removed: Vec<String> = {
            let mut map = self.map.write().expect("alias lock poisoned");
            let names: Vec<String> =
                map.values().filter(|a| a.managed && a.model == model).map(|a| a.name.clone()).collect();
            for n in &names {
                map.remove(n);
            }
            names
        };
        if !removed.is_empty() {
            self.save();
        }
        removed
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        let _saving = self.saving.lock().expect("alias mutex poisoned");
        let managed: Vec<Alias> = self
            .list()
            .into_iter()
            .filter(|a| a.managed)
            .map(|a| Alias { name: a.name, model: a.model, upstream_model: a.upstream_model })
            .collect();
        if let Err(e) = write_json(path, &managed) {
            tracing::error!("saving {}: {e}", path.display());
        }
    }
}

fn view(a: Alias, managed: bool) -> AliasView {
    AliasView { name: a.name, model: a.model, upstream_model: a.upstream_model, managed }
}

fn error(e: AliasError) -> Response {
    match e {
        AliasError::Invalid(m) => bad_request(&m),
        AliasError::NotFound(m) => not_found(&m),
        AliasError::Conflict(m) | AliasError::InConfig(m) => {
            (StatusCode::CONFLICT, Json(json!({"error": {"message": m, "type": "conflict"}}))).into_response()
        }
    }
}

pub async fn list(State(state): State<AppState>) -> Response {
    Json(state.aliases.list()).into_response()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasBody {
    pub model: String,
    #[serde(default)]
    pub upstream_model: Option<String>,
}

/// `PUT /api/aliases/{name}`: create or retarget.
pub async fn put(State(state): State<AppState>, UrlPath(name): UrlPath<String>, Json(b): Json<AliasBody>) -> Response {
    let alias = Alias { name, model: b.model, upstream_model: b.upstream_model.filter(|u| !u.is_empty()) };
    match state.aliases.set(alias, &state.manager) {
        Ok(a) => Json(a).into_response(),
        Err(e) => error(e),
    }
}

pub async fn delete(State(state): State<AppState>, UrlPath(name): UrlPath<String>) -> Response {
    match state.aliases.remove(&name) {
        Ok(a) => Json(a).into_response(),
        Err(e) => error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use contio_backends::{ManagerConfig, ModelManager};
    use contio_core::hw::DeviceKind;
    use contio_core::model::{ModelProfile, ModelSource};

    fn manager(ids: &[&str]) -> ModelManager {
        let profiles = ids
            .iter()
            .map(|id| ModelProfile::new(*id, ModelSource::Flm { tag: (*id).into() }, DeviceKind::Npu))
            .collect();
        ModelManager::new(
            ManagerConfig {
                log_dir: std::env::temp_dir(),
                startup_timeout: std::time::Duration::from_secs(1),
                default_idle_ttl: None,
                pool_budgets: Default::default(),
            },
            vec![],
            profiles,
        )
    }

    fn alias(name: &str, model: &str) -> Alias {
        Alias { name: name.into(), model: model.into(), upstream_model: None }
    }

    #[test]
    fn api_aliases_are_validated_and_saved() {
        let dir = std::env::temp_dir().join(format!("contio-alias-test-{}", std::process::id()));
        let path = dir.join("aliases.json");
        let m = manager(&["a", "b"]);
        let aliases = Aliases::new(vec![alias("fixed", "a")], Some(path.clone()), &m);

        assert!(matches!(aliases.set(alias("a", "b"), &m), Err(AliasError::Conflict(_))));
        assert!(matches!(aliases.set(alias("x", "nope"), &m), Err(AliasError::NotFound(_))));
        assert!(matches!(aliases.set(alias("Bad Name", "a"), &m), Err(AliasError::Invalid(_))));
        assert!(matches!(aliases.set(alias("fixed", "b"), &m), Err(AliasError::InConfig(_))));
        assert!(matches!(aliases.remove("fixed"), Err(AliasError::InConfig(_))));

        aliases.set(alias("coder", "a"), &m).unwrap();
        aliases.set(alias("coder", "b"), &m).unwrap(); // retarget
        aliases.set(alias("chat", "b"), &m).unwrap();
        assert_eq!(aliases.get("coder").unwrap().model, "b");

        // reload: only API aliases come from the file
        let reloaded = Aliases::new(vec![], Some(path.clone()), &m);
        let names: Vec<String> = reloaded.list().into_iter().map(|a| a.name).collect();
        assert_eq!(names, ["chat", "coder"]);

        assert_eq!(reloaded.remove_for_model("b"), ["chat", "coder"]);
        assert!(Aliases::new(vec![], Some(path), &m).list().is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
}
