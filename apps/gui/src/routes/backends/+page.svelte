<script lang="ts">
  import { checkPreflight, errorText, fetchBackends, installBackend, saveSettings } from "$lib/daemon";
  import { jobs, refreshJobs } from "$lib/jobs.svelte";
  import { live } from "$lib/live.svelte";
  import type { BackendId, BackendsInfo, InstallMethod, PreflightResult } from "$lib/types";

  let info = $state<BackendsInfo | null>(null);
  let error = $state<string | null>(null);
  let checks = $state<Partial<Record<BackendId, PreflightResult>>>({});
  let jobIds = $state<Partial<Record<BackendId, number>>>({});
  let copied = $state<string | null>(null);

  const BACKENDS: {
    id: BackendId;
    name: string;
    what: string;
    registeredAs: string;
    methods: { id: InstallMethod; name: string; note: string }[];
  }[] = [
    {
      id: "llama.cpp",
      name: "llama.cpp",
      what: "Runs GGUF models on the GPU (Vulkan) or CPU. Needed for almost every catalog model.",
      registeredAs: "llama.cpp",
      methods: [
        { id: "prebuilt", name: "Prebuilt", note: "official release binary, ~35 MB, ready in seconds (recommended)" },
        { id: "source", name: "Build from source", note: "tuned for this CPU; needs compilers and Vulkan headers, ~2 min" },
        { id: "podman", name: "Podman container", note: "official image, nothing installed on the host" },
      ],
    },
    {
      id: "flm",
      name: "FastFlowLM",
      what: "Runs models on the Ryzen AI NPU at a few watts, next to GPU models.",
      registeredAs: "FastFlowLM",
      methods: [{ id: "source", name: "Build from source", note: "the only option; needs the XRT runtime and build tools" }],
    },
    {
      id: "whisper.cpp",
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

  const method = (b: BackendId): InstallMethod => info?.settings[SETTING[b]] ?? "source";

  async function setMethod(b: BackendId, m: InstallMethod) {
    if (!info) return;
    const next = { ...info.settings, [SETTING[b]]: m };
    try {
      info.settings = await saveSettings(next);
      delete checks[b];
    } catch (e) {
      error = errorText(e);
    }
  }

  async function check(b: BackendId) {
    try {
      checks[b] = await checkPreflight(b, method(b));
    } catch (e) {
      error = errorText(e);
    }
  }

  async function install(b: BackendId) {
    try {
      jobIds[b] = (await installBackend(b, method(b))).job_id;
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
            <td>{i.backend}</td>
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
</style>
