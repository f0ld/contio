<script lang="ts">
  import { checkPreflight, errorText, fetchBackends, installBackend, saveSettings } from "$lib/daemon";
  import { jobs, refreshJobs } from "$lib/jobs.svelte";
  import { live } from "$lib/live.svelte";
  import type { BackendId, BackendsInfo, GpuApi, InstallMethod, PreflightResult } from "$lib/types";

  /** One card: a backend, or the optional ROCm build of llama.cpp. */
  type CardId = BackendId | "llama.cpp-rocm";

  let info = $state<BackendsInfo | null>(null);
  let error = $state<string | null>(null);
  let checks = $state<Partial<Record<CardId, PreflightResult>>>({});
  let jobIds = $state<Partial<Record<CardId, number>>>({});
  let copied = $state<string | null>(null);
  // the ROCm build has no saved method setting: it's only installed from here
  let rocmMethod = $state<InstallMethod>("source");

  const BACKENDS: {
    id: CardId;
    backend: BackendId;
    gpu?: GpuApi;
    name: string;
    what: string;
    registeredAs: string;
    optional?: boolean;
    methods: { id: InstallMethod; name: string; note: string }[];
  }[] = [
    {
      id: "llama.cpp",
      backend: "llama.cpp",
      name: "llama.cpp",
      what: "Runs GGUF models on the GPU (Vulkan) or CPU. Needed for almost every catalog model.",
      registeredAs: "llama.cpp Vulkan",
      methods: [
        { id: "prebuilt", name: "Prebuilt", note: "official release binary, ~35 MB, ready in seconds (recommended)" },
        { id: "source", name: "Build from source", note: "tuned for this CPU; needs compilers and Vulkan headers, ~2 min" },
        { id: "podman", name: "Podman container", note: "official image, nothing installed on the host" },
      ],
    },
    {
      id: "llama.cpp-rocm",
      backend: "llama.cpp",
      gpu: "rocm",
      name: "llama.cpp for ROCm",
      what:
        "Optional second GPU build. On the Radeon 890M it reads long prompts 3–9% faster than Vulkan but writes answers 5–8% slower. Pick it per model, or as the default below.",
      registeredAs: "llama.cpp ROCm",
      optional: true,
      methods: [
        { id: "source", name: "Build from source", note: "needs the ROCm/HIP development packages, ~4 min" },
        { id: "podman", name: "Podman container", note: "official ROCm image, ~23 GB download, nothing installed on the host" },
      ],
    },
    {
      id: "flm",
      backend: "flm",
      name: "FastFlowLM",
      what: "Runs models on the Ryzen AI NPU at a few watts, next to GPU models.",
      registeredAs: "FastFlowLM",
      methods: [{ id: "source", name: "Build from source", note: "the only option; needs the XRT runtime and build tools" }],
    },
    {
      id: "whisper.cpp",
      backend: "whisper.cpp",
      name: "whisper.cpp",
      what: "Speech to text with Whisper models on the GPU (Vulkan) or CPU.",
      registeredAs: "whisper.cpp",
      methods: [{ id: "source", name: "Build from source", note: "the only option on Linux; same build tools as llama.cpp, ~2 min" }],
    },
  ];

  async function load() {
    try {
      info = await fetchBackends();
      error = null;
    } catch (e) {
      error = errorText(e);
    }
  }

  $effect(() => {
    if (live.online && !info) load();
  });

  // reload when a backend job finishes
  $effect(() => {
    for (const id of Object.values(jobIds)) {
      if (jobs.byId(id ?? null)?.state === "done") load();
    }
  });

  const SETTING = { "llama.cpp": "llamacpp_method", flm: "flm_method", "whisper.cpp": "whisper_method" } as const;

  const card = (id: CardId) => BACKENDS.find((b) => b.id === id)!;
  const method = (id: CardId): InstallMethod =>
    id === "llama.cpp-rocm" ? rocmMethod : (info?.settings[SETTING[id]] ?? "source");

  async function setMethod(id: CardId, m: InstallMethod) {
    delete checks[id];
    if (id === "llama.cpp-rocm") {
      rocmMethod = m;
      return;
    }
    if (!info) return;
    const next = { ...info.settings, [SETTING[id]]: m };
    try {
      info.settings = await saveSettings(next);
    } catch (e) {
      error = errorText(e);
    }
  }

  async function setDefaultGpu(api: GpuApi) {
    if (!info) return;
    try {
      info.settings = await saveSettings({ ...info.settings, gpu_api: api });
      error = null;
    } catch (e) {
      error = errorText(e);
    }
  }

  async function check(id: CardId) {
    try {
      checks[id] = await checkPreflight(card(id).backend, method(id), card(id).gpu);
    } catch (e) {
      error = errorText(e);
    }
  }

  async function install(id: CardId) {
    try {
      jobIds[id] = (await installBackend(card(id).backend, method(id), card(id).gpu)).job_id;
      refreshJobs();
    } catch (e) {
      error = errorText(e);
    }
  }

  async function copy(text: string) {
    await navigator.clipboard.writeText(text);
    copied = text;
    setTimeout(() => (copied = null), 1500);
  }

  const registered = (name: string) => info?.registered.filter((r) => r.startsWith(name)) ?? [];
</script>

{#if error}<p class="error-text">{error}</p>{/if}

<p class="sub intro">
  Backends are set up automatically the first time a model needs one, with the method chosen here. You can also
  install them now.
</p>

{#each BACKENDS as b (b.id)}
  {@const job = jobs.byId(jobIds[b.id] ?? null)}
  {@const pf = checks[b.id]}
  <section class="card">
    <div class="head">
      <h2>{b.name}</h2>
      {#if registered(b.registeredAs).length}
        <span class="pill fits">ready</span>
      {:else if b.optional}
        <span class="pill">optional</span>
      {:else}
        <span class="pill tight">not set up</span>
      {/if}
    </div>
    <p class="sub">{b.what}</p>
    {#each registered(b.registeredAs) as r (r)}<div class="mono">{r}</div>{/each}

    <div class="methods">
      {#each b.methods as m (m.id)}
        <label class="method">
          <input type="radio" name="method-{b.id}" checked={method(b.id) === m.id} onchange={() => setMethod(b.id, m.id)} />
          <b>{m.name}</b> <span class="sub">{m.note}</span>
        </label>
      {/each}
    </div>

    <div class="actions">
      <button onclick={() => check(b.id)}>Check prerequisites</button>
      <button class="primary" disabled={job?.state === "running"} onclick={() => install(b.id)}>
        {registered(b.registeredAs).length ? "Reinstall" : "Install now"}
      </button>
      {#if job}
        <span class="sub">
          job #{job.id}: {job.state === "running" ? job.step ?? "starting" : job.state}
          · <a href="/jobs">details</a>
        </span>
      {/if}
    </div>
    {#if job?.error}<p class="error-text">{job.error}</p>{/if}

    {#if b.id === "llama.cpp-rocm" && info}
      {@const rocmReady = registered("llama.cpp ROCm").length > 0}
      <fieldset class="gpu-default">
        <legend>Default GPU backend for GGUF models</legend>
        <label class="method">
          <input type="radio" name="gpu-default" checked={info.settings.gpu_api === "vulkan"} onchange={() => setDefaultGpu("vulkan")} />
          <b>Vulkan</b> <span class="sub">faster answers; the better pick for most use (recommended)</span>
        </label>
        <label class="method" title={rocmReady ? "" : "Install the ROCm build first"}>
          <input
            type="radio"
            name="gpu-default"
            disabled={!rocmReady && info.settings.gpu_api !== "rocm"}
            checked={info.settings.gpu_api === "rocm"}
            onchange={() => setDefaultGpu("rocm")}
          />
          <b>ROCm</b> <span class="sub">faster on long prompts{rocmReady ? "" : " · install it first"}</span>
        </label>
        <p class="sub">Models can override this in their settings. Loaded models restart on their next request.</p>
      </fieldset>
    {/if}

    {#if pf}
      {#if pf.ok}
        <p class="ok">✓ All prerequisites are installed.</p>
      {:else}
        <div class="missing">
          {#if pf.preflight.missing_packages.length}
            <p>Missing packages: {pf.preflight.missing_packages.join(", ")}</p>
          {/if}
          {#if pf.preflight.missing_tools.length}<p>Missing tools: {pf.preflight.missing_tools.join(", ")}</p>{/if}
          {#each pf.preflight.notes as n (n)}<p>{n}</p>{/each}
          {#if pf.install_hint}
            <p class="sub">Run this in a terminal (contio never runs root commands itself), then check again:</p>
            <pre class="cmd">{pf.install_hint}</pre>
            <button class="small" onclick={() => copy(pf.install_hint ?? "")}>
              {copied === pf.install_hint ? "Copied" : "Copy command"}
            </button>
          {/if}
        </div>
      {/if}
    {/if}
  </section>
{/each}

{#if info?.installed.length}
  <section class="card">
    <h2>Installed by contio</h2>
    <table class="list">
      <thead><tr><th>Backend</th><th>Version</th><th>Method</th><th>Location</th></tr></thead>
      <tbody>
        {#each info.installed as i (i.dir)}
          <tr>
            <td>{i.backend}{i.backend === "llama.cpp" ? (i.gpu_api === "rocm" ? " ROCm" : " Vulkan") : ""}</td>
            <td>{i.version.slice(0, 12)}</td>
            <td>{i.method}</td>
            <td class="mono sub">{i.bin_dir ?? i.image ?? i.prefix}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="sub">Backends set in <code>config.toml</code> take precedence over installed ones.</p>
  </section>
{/if}

<style>
  .intro {
    margin: 0;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .head h2 {
    margin: 0;
  }
  .mono {
    font-family: ui-monospace, monospace;
    font-size: 0.8rem;
    word-break: break-all;
  }
  .methods {
    display: grid;
    gap: 0.35rem;
    margin: 0.75rem 0;
  }
  .method {
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .ok {
    color: var(--ok);
  }
  .missing p {
    margin: 0.4rem 0;
  }
  .gpu-default {
    border: 1px solid var(--border);
    border-radius: 6px;
    margin: 0.9rem 0 0;
    padding: 0.5rem 0.8rem 0.6rem;
    display: grid;
    gap: 0.35rem;
  }
  .gpu-default legend {
    font-weight: 600;
    padding: 0 0.3rem;
  }
  .gpu-default p {
    margin: 0.2rem 0 0;
  }
</style>
