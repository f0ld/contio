//! llama.cpp `llama-server` (GGUF on GPU via Vulkan or ROCm, or CPU), run
//! natively or in a Podman container.

use std::path::{Path, PathBuf};

use contio_core::hw::{DeviceKind, GpuApi};
use contio_core::model::{ModelKind, ModelProfile, ModelSource, estimate_memory};
use tokio::process::Command;

use crate::{Backend, BackendError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaRuntime {
    /// Directory containing `llama-server` (a GPU build also serves CPU).
    Native { bin_dir: PathBuf },
    /// A llama.cpp server image (entrypoint `llama-server`), run with `podman run`.
    Podman { image: String },
}

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub runtime: LlamaRuntime,
    /// What the build uses for GPU models.
    pub gpu_api: GpuApi,
    /// Threads for CPU inference (physical cores work best, see m0-gpu.md).
    pub cpu_threads: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct LlamaCppBackend {
    config: LlamaCppConfig,
}

impl LlamaCppBackend {
    pub fn new(config: LlamaCppConfig) -> Self {
        Self { config }
    }

    fn gguf_path(profile: &ModelProfile) -> Option<&PathBuf> {
        match &profile.source {
            ModelSource::Gguf { path, .. } => Some(path),
            ModelSource::Flm { .. } | ModelSource::Whisper { .. } => None,
        }
    }

    fn mmproj_path(profile: &ModelProfile) -> Option<&PathBuf> {
        match &profile.source {
            ModelSource::Gguf { mmproj, .. } => mmproj.as_ref(),
            ModelSource::Flm { .. } | ModelSource::Whisper { .. } => None,
        }
    }

    /// `--spec-type` value for the profile; MTP only if the file has the layers.
    fn spec_types(profile: &ModelProfile) -> Option<String> {
        let s = profile.speculative;
        let mtp = s.mtp && Self::gguf_path(profile).is_some_and(|p| crate::gguf::has_mtp(p));
        if s.mtp && !mtp {
            tracing::warn!(model = %profile.id, "MTP requested, but the model file has no MTP layers");
        }
        // n-gram first: draftless methods take precedence where they have a guess (docs/speculative.md)
        let types: Vec<&str> = [(s.ngram, "ngram-simple"), (mtp, "draft-mtp")]
            .into_iter()
            .filter_map(|(on, t)| on.then_some(t))
            .collect();
        (!types.is_empty()).then(|| types.join(","))
    }

    /// `llama-server` arguments; paths and `host` differ between runtimes.
    fn args(&self, profile: &ModelProfile, port: u16, model: &str, mmproj: Option<&str>, host: &str) -> Vec<String> {
        let mut a: Vec<String> = vec![
            "--model".into(),
            model.to_owned(),
            "--host".into(),
            host.to_owned(),
            "--port".into(),
            port.to_string(),
            "--alias".into(),
            profile.id.clone(),
            // chat templates from the GGUF: needed for tool calling (m0-gpu.md)
            "--jinja".into(),
            "--flash-attn".into(),
            "on".into(),
        ];
        if let Some(ctx) = profile.context {
            a.extend(["--ctx-size".into(), ctx.to_string()]);
        }
        if let Some(m) = mmproj {
            a.extend(["--mmproj".into(), m.to_owned()]);
        }
        match profile.kind {
            ModelKind::Chat => {}
            kind => {
                a.push(if kind == ModelKind::Embedding { "--embedding" } else { "--reranking" }.into());
                // non-causal models need a whole input in one micro-batch
                let batch = profile.context.unwrap_or(8192).to_string();
                a.extend(["--batch-size".into(), batch.clone(), "--ubatch-size".into(), batch]);
            }
        }
        match profile.device {
            DeviceKind::Gpu => a.extend(["--n-gpu-layers".into(), "999".into()]),
            _ => {
                a.extend(["--n-gpu-layers".into(), "0".into(), "--device".into(), "none".into()]);
                if let Some(t) = self.config.cpu_threads {
                    a.extend(["--threads".into(), t.to_string()]);
                }
            }
        }
        if profile.kind == ModelKind::Chat
            && let Some(types) = Self::spec_types(profile)
        {
            a.extend(["--spec-type".into(), types]);
        }
        a.extend(profile.extra_args.iter().cloned());
        a
    }

    /// `podman run` arguments: model directory mounted read-only at /models
    /// (the projector's at /mmproj if elsewhere), port published on localhost
    /// only, GPU via /dev/dri.
    fn podman_args(&self, image: &str, profile: &ModelProfile, port: u16, model: &Path) -> Vec<String> {
        let parent = |p: &Path| p.parent().unwrap_or(Path::new("/")).to_owned();
        let file_name = |p: &Path| p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let dir = parent(model);
        let file = file_name(model);
        let name: String =
            profile.id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect();
        let mut a: Vec<String> = vec![
            "run".into(),
            "--rm".into(),
            "--name".into(),
            format!("contio-{name}-{port}"),
            // Fedora: SELinux would block /dev/dri and the bind mount otherwise
            "--security-opt".into(),
            "label=disable".into(),
            "--publish".into(),
            format!("127.0.0.1:{port}:{port}"),
            "--volume".into(),
            format!("{}:/models:ro", dir.display()),
        ];
        let mmproj = Self::mmproj_path(profile).map(|m| {
            if parent(m) == dir {
                format!("/models/{}", file_name(m))
            } else {
                a.extend(["--volume".into(), format!("{}:/mmproj:ro", parent(m).display())]);
                format!("/mmproj/{}", file_name(m))
            }
        });
        if profile.device == DeviceKind::Gpu {
            a.extend(["--device".into(), "/dev/dri".into()]);
            if self.config.gpu_api == GpuApi::Rocm {
                a.extend(["--device".into(), "/dev/kfd".into()]);
            }
        }
        a.push(image.to_owned());
        a.extend(self.args(profile, port, &format!("/models/{file}"), mmproj.as_deref(), "0.0.0.0"));
        a
    }
}

/// Size of a GGUF model on disk, all shards of a split model included
/// (`name-00001-of-00003.gguf` → shards 1 to 3 in the same directory).
pub(crate) fn gguf_bytes(path: &Path) -> Option<u64> {
    let first = std::fs::metadata(path).ok()?.len();
    let name = path.file_name()?.to_str()?;
    let Some((prefix, total)) = split_name(name) else { return Some(first) };
    let mut sum = first;
    for i in 2..=total {
        sum += std::fs::metadata(path.with_file_name(format!("{prefix}-{i:05}-of-{total:05}.gguf"))).ok()?.len();
    }
    Some(sum)
}

/// `("name", 3)` for `name-00001-of-00003.gguf`.
fn split_name(name: &str) -> Option<(&str, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = rest.rsplit_once('-')?;
    (index == "00001" && total.len() == 5).then_some((prefix, total.parse().ok()?))
}

/// First `name` on `$PATH`.
pub(crate) fn find_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_str()?
        .split(':')
        .map(|d| Path::new(d).join(name))
        .find(|p| p.is_file())
}

impl Backend for LlamaCppBackend {
    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    fn describe(&self) -> String {
        let api = self.config.gpu_api;
        match &self.config.runtime {
            LlamaRuntime::Native { bin_dir } => format!("llama.cpp {api} (native: {})", bin_dir.display()),
            LlamaRuntime::Podman { image } => format!("llama.cpp {api} (podman: {image})"),
        }
    }

    fn gpu_api(&self) -> Option<GpuApi> {
        Some(self.config.gpu_api)
    }

    /// GPU models only on a build for their API (unset: Vulkan); any build runs CPU models.
    fn supports(&self, profile: &ModelProfile) -> bool {
        Self::gguf_path(profile).is_some()
            && match profile.device {
                DeviceKind::Gpu => profile.gpu_api.unwrap_or_default() == self.config.gpu_api,
                DeviceKind::Cpu => true,
                _ => false,
            }
    }

    fn command(&self, profile: &ModelProfile, port: u16) -> Result<Command, BackendError> {
        let path = Self::gguf_path(profile).ok_or_else(|| BackendError::Unsupported(profile.id.clone()))?;
        for file in std::iter::once(path).chain(Self::mmproj_path(profile)) {
            if !file.is_file() {
                return Err(BackendError::MissingModelFile(file.clone()));
            }
        }
        match &self.config.runtime {
            LlamaRuntime::Native { bin_dir } => {
                let bin = bin_dir.join("llama-server");
                if !bin.is_file() {
                    return Err(BackendError::MissingBinary(bin));
                }
                let mut cmd = Command::new(bin);
                let mmproj = Self::mmproj_path(profile).map(|m| m.display().to_string());
                cmd.args(self.args(profile, port, &path.display().to_string(), mmproj.as_deref(), "127.0.0.1"));
                Ok(cmd)
            }
            LlamaRuntime::Podman { image } => {
                let podman = find_in_path("podman").ok_or_else(|| BackendError::MissingBinary("podman".into()))?;
                let mut cmd = Command::new(podman);
                cmd.args(self.podman_args(image, profile, port, path));
                Ok(cmd)
            }
        }
    }

    fn health_path(&self) -> &'static str {
        "/health"
    }

    fn upstream_model(&self, profile: &ModelProfile) -> String {
        profile.id.clone() // served under --alias
    }

    fn estimate_bytes(&self, profile: &ModelProfile) -> Option<u64> {
        let mut weights = gguf_bytes(Self::gguf_path(profile)?)?;
        if let Some(m) = Self::mmproj_path(profile) {
            weights += std::fs::metadata(m).map(|m| m.len()).unwrap_or(0);
        }
        Some(estimate_memory(weights, profile.kv.as_ref(), profile.context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(device: DeviceKind) -> ModelProfile {
        let mut p = ModelProfile::new("qwen3-8b", ModelSource::Gguf { path: "/m/q.gguf".into(), mmproj: None }, device);
        p.context = Some(16384);
        p.extra_args = vec!["--cache-type-k".into(), "q8_0".into()];
        p
    }

    fn backend() -> LlamaCppBackend {
        LlamaCppBackend::new(LlamaCppConfig {
            runtime: LlamaRuntime::Native { bin_dir: "/nonexistent".into() },
            gpu_api: GpuApi::Vulkan,
            cpu_threads: Some(12),
        })
    }

    fn rocm_backend() -> LlamaCppBackend {
        LlamaCppBackend::new(LlamaCppConfig {
            runtime: LlamaRuntime::Podman { image: "img".into() },
            gpu_api: GpuApi::Rocm,
            cpu_threads: None,
        })
    }

    fn native_args(p: &ModelProfile) -> String {
        backend().args(p, 40001, "/m/q.gguf", None, "127.0.0.1").join(" ")
    }

    #[test]
    fn gpu_args() {
        let a = native_args(&profile(DeviceKind::Gpu));
        assert!(a.starts_with("--model /m/q.gguf --host 127.0.0.1 --port 40001 --alias qwen3-8b --jinja"));
        assert!(a.contains("--ctx-size 16384"));
        assert!(a.contains("--n-gpu-layers 999"));
        assert!(!a.contains("--threads"));
        assert!(a.ends_with("--cache-type-k q8_0"));
    }

    #[test]
    fn cpu_args_disable_gpu_and_set_threads() {
        let a = native_args(&profile(DeviceKind::Cpu));
        assert!(a.contains("--n-gpu-layers 0 --device none --threads 12"));
    }

    #[test]
    fn supports_only_gguf_on_cpu_gpu() {
        let b = backend();
        assert!(b.supports(&profile(DeviceKind::Gpu)));
        assert!(!b.supports(&profile(DeviceKind::Npu)));
        let mut flm = profile(DeviceKind::Gpu);
        flm.source = ModelSource::Flm { tag: "x".into() };
        assert!(!b.supports(&flm));
    }

    #[test]
    fn gpu_models_need_a_build_for_their_api() {
        let (vulkan, rocm) = (backend(), rocm_backend());
        let mut p = profile(DeviceKind::Gpu);
        assert!(vulkan.supports(&p) && !rocm.supports(&p), "unset means Vulkan");
        p.gpu_api = Some(GpuApi::Rocm);
        assert!(!vulkan.supports(&p) && rocm.supports(&p));
        let cpu = profile(DeviceKind::Cpu);
        assert!(vulkan.supports(&cpu) && rocm.supports(&cpu), "either build runs CPU models");
        assert!(rocm.describe().starts_with("llama.cpp ROCm (podman"));
        let a = rocm.podman_args("img", &p, 1, Path::new("/m/q.gguf")).join(" ");
        assert!(a.contains("--device /dev/dri --device /dev/kfd"), "{a}");
        assert!(!backend().podman_args("img", &p, 1, Path::new("/m/q.gguf")).join(" ").contains("kfd"));
    }

    #[test]
    fn missing_files_are_reported() {
        let err = backend().command(&profile(DeviceKind::Gpu), 1).unwrap_err();
        assert!(matches!(err, BackendError::MissingModelFile(_)), "{err}");

        let model = tempfile::NamedTempFile::new().unwrap();
        let mut p = profile(DeviceKind::Gpu);
        p.source = ModelSource::Gguf { path: model.path().into(), mmproj: Some("/missing/mmproj.gguf".into()) };
        let err = backend().command(&p, 1).unwrap_err();
        assert!(matches!(err, BackendError::MissingModelFile(ref f) if f.ends_with("mmproj.gguf")), "{err}");
        p.source = ModelSource::Gguf { path: model.path().into(), mmproj: None };
        let err = backend().command(&p, 1).unwrap_err();
        assert!(matches!(err, BackendError::MissingBinary(_)), "{err}");
    }

    #[test]
    fn podman_args_mount_model_and_publish_localhost() {
        let mut p = profile(DeviceKind::Gpu);
        p.id = "qwen3.6:35b".into();
        let a = backend()
            .podman_args("ghcr.io/ggml-org/llama.cpp:server-vulkan", &p, 40001, Path::new("/data/gguf/q.gguf"))
            .join(" ");
        assert!(a.starts_with("run --rm --name contio-qwen3-6-35b-40001 --security-opt label=disable"));
        assert!(a.contains("--publish 127.0.0.1:40001:40001 --volume /data/gguf:/models:ro --device /dev/dri"));
        assert!(a.contains("ghcr.io/ggml-org/llama.cpp:server-vulkan --model /models/q.gguf --host 0.0.0.0 --port 40001"));

        let cpu = backend().podman_args("img", &profile(DeviceKind::Cpu), 1, Path::new("/m/q.gguf")).join(" ");
        assert!(!cpu.contains("/dev/dri"));

        p.source = ModelSource::Gguf { path: "/data/gguf/q.gguf".into(), mmproj: Some("/data/gguf/mmproj-F16.gguf".into()) };
        let a = backend().podman_args("img", &p, 1, Path::new("/data/gguf/q.gguf")).join(" ");
        assert!(a.contains("--mmproj /models/mmproj-F16.gguf") && !a.contains("/mmproj:ro"));
        p.source = ModelSource::Gguf { path: "/data/gguf/q.gguf".into(), mmproj: Some("/other/mm.gguf".into()) };
        let a = backend().podman_args("img", &p, 1, Path::new("/data/gguf/q.gguf")).join(" ");
        assert!(a.contains("--volume /other:/mmproj:ro") && a.contains("--mmproj /mmproj/mm.gguf"));
    }

    #[test]
    fn speculative_decoding_args() {
        let d = tempfile::tempdir().unwrap();
        let (mtp_file, plain) = (d.path().join("mtp.gguf"), d.path().join("plain.gguf"));
        crate::gguf::tests::write_gguf(&mtp_file, Some(1));
        crate::gguf::tests::write_gguf(&plain, None);
        let mut p = profile(DeviceKind::Gpu);
        p.source = ModelSource::Gguf { path: mtp_file, mmproj: None };
        assert!(!native_args(&p).contains("--spec-type"), "off by default");
        p.speculative.ngram = true;
        assert!(native_args(&p).contains("--spec-type ngram-simple --cache-type-k"), "before extra args");
        p.speculative.mtp = true;
        assert!(native_args(&p).contains("--spec-type ngram-simple,draft-mtp"));
        p.source = ModelSource::Gguf { path: plain, mmproj: None };
        assert!(native_args(&p).contains("--spec-type ngram-simple "), "no MTP layers: MTP is skipped");
        p.speculative.ngram = false;
        assert!(!native_args(&p).contains("--spec-type"));
        p.speculative = contio_core::model::Speculative { ngram: true, mtp: false };
        p.kind = ModelKind::Embedding;
        assert!(!native_args(&p).contains("--spec-type"), "chat models only");
    }

    #[test]
    fn embedding_and_reranker_modes() {
        let mut p = profile(DeviceKind::Gpu);
        p.kind = ModelKind::Embedding;
        p.context = Some(2048);
        let a = native_args(&p);
        assert!(a.contains("--embedding --batch-size 2048 --ubatch-size 2048"), "{a}");
        p.kind = ModelKind::Reranker;
        assert!(native_args(&p).contains("--reranking"));
        assert!(!native_args(&profile(DeviceKind::Gpu)).contains("--embedding"));
    }

    #[test]
    fn split_models_are_summed() {
        let d = tempfile::tempdir().unwrap();
        for (name, len) in [("m-00001-of-00002.gguf", 10), ("m-00002-of-00002.gguf", 5), ("single.gguf", 7)] {
            std::fs::write(d.path().join(name), vec![0u8; len]).unwrap();
        }
        assert_eq!(gguf_bytes(&d.path().join("m-00001-of-00002.gguf")), Some(15));
        assert_eq!(gguf_bytes(&d.path().join("single.gguf")), Some(7));
        std::fs::remove_file(d.path().join("m-00002-of-00002.gguf")).unwrap();
        assert_eq!(gguf_bytes(&d.path().join("m-00001-of-00002.gguf")), None, "incomplete download");
    }
}
