<script lang="ts">
  import { deleteAlias, errorText, fetchBackends, formatContext, resetOverrides, setAlias, updateModel } from "$lib/daemon";
  import { PERF_SETTINGS } from "$lib/performance";
  import type { Alias, BackendsInfo, GpuApi, ModelPatch, ModelStatus } from "$lib/types";

  /** Launch settings and aliases of one model. Changes to config.toml models are saved as overrides. */
  let {
    model,
    aliases,
    onchange,
    onclose,
  }: { model: ModelStatus; aliases: Alias[]; onchange: () => void; onclose: () => void } = $props();

  // the form starts from the model as it was when the editor opened
  // svelte-ignore state_referenced_locally
  const initial = model;
  let context = $state<number | null>(initial.context);
  let device = $state(initial.device);
  let pinned = $state(initial.pinned);
  let ttlMinutes = $state<number | null>(initial.idle_ttl_secs ? Math.round(initial.idle_ttl_secs / 60) : null);
  let extraArgs = $state(initial.extra_args.join(" "));
  let gpuApi = $state<GpuApi | "default">(initial.gpu_api ?? "default");
  let backends = $state<BackendsInfo | null>(null);
  const isGguf = initial.source.format === "gguf";
  if (isGguf) fetchBackends().then((b) => (backends = b), () => {});
  const hasBuild = (api: GpuApi) => backends?.registered.some((r) => r.startsWith(`llama.cpp ${api === "rocm" ? "ROCm" : "Vulkan"}`));
  const API_NAME = { vulkan: "Vulkan", rocm: "ROCm" } as const;
  let ngram = $state(initial.speculative?.ngram ?? false);
  let mtp = $state(initial.speculative?.mtp ?? false);
  const speculative = isGguf && initial.kind === "chat";
  const hasExpert = isGguf;
  let saving = $state(false);
  let error = $state<string | null>(null);
  let newAlias = $state("");

  let mine = $derived(aliases.filter((a) => a.model === model.id));
  const movable = initial.source.format !== "flm";
  const file = initial.source.format === "flm" ? `FastFlowLM ${initial.source.tag}` : initial.source.path;

  function changes(): ModelPatch {
    const p: ModelPatch = {};
    if (context && context !== initial.context) p.context = context;
    if (device !== initial.device && (device === "gpu" || device === "cpu")) p.device = device;
    if (pinned !== initial.pinned) p.pinned = pinned;
    const ttl = ttlMinutes ? ttlMinutes * 60 : 0;
    if (ttl !== (initial.idle_ttl_secs ?? 0)) p.idle_ttl_secs = ttl;
    const args = extraArgs.split(/\s+/).filter(Boolean);
    if (args.join(" ") !== initial.extra_args.join(" ")) p.extra_args = args;
    if (gpuApi !== (initial.gpu_api ?? "default")) p.gpu_api = gpuApi;
    if (ngram !== (initial.speculative?.ngram ?? false) || mtp !== (initial.speculative?.mtp ?? false))
      p.speculative = { ngram, mtp };
    return p;
  }
  let dirty = $derived(Object.keys(changes()).length > 0);

  async function save() {
    saving = true;
    error = null;
    try {
      await updateModel(model.id, changes());
      onchange();
      onclose();
    } catch (e) {
      error = errorText(e);
    } finally {
      saving = false;
    }
  }

  async function reset() {
    saving = true;
    error = null;
    try {
      await resetOverrides(model.id);
      onchange();
      onclose();
    } catch (e) {
      error = errorText(e);
    } finally {
      saving = false;
    }
  }

  async function aliasAction(f: () => Promise<unknown>) {
    error = null;
    try {
      await f();
      newAlias = "";
      onchange();
    } catch (e) {
      error = errorText(e);
    }
  }
</script>

<div class="editor">
  <div class="file sub" title={file}>{file}</div>

  {#if !model.managed}
    <p class="sub">
      From <code>config.toml</code>. Changes here are saved as overrides on top of it; the file stays as it is.
    </p>
  {/if}
  <div class="grid">
    {#if model.kind !== "transcription"}
      <label>
        Context
        <input type="number" min="512" step="1024" bind:value={context} placeholder="backend default" />
        <span class="sub">{formatContext(context)}</span>
      </label>
    {/if}
    <label>
      Device
      {#if movable}
        <select bind:value={device}>
          <option value="gpu">GPU</option>
          <option value="cpu">CPU</option>
        </select>
      {:else}
        <span>{model.device.toUpperCase()}</span>
      {/if}
    </label>
    <label title="Unload after this many idle minutes; empty or 0 = daemon default">
      Idle unload
      <input type="number" min="0" step="5" bind:value={ttlMinutes} placeholder="default" />
      <span class="sub">min</span>
    </label>
    <label title="Never unloaded for being idle or evicted for other models">
      <input type="checkbox" bind:checked={pinned} /> pinned
    </label>
    <label class="wide">
      Extra backend arguments
      <input class="mono" bind:value={extraArgs} placeholder="e.g. --temp 0.6 --top-k 20" />
    </label>
  </div>

  {#if hasExpert}
    <details class="expert" open={initial.speculative?.ngram || initial.speculative?.mtp || initial.gpu_api !== null}>
      <summary><span class="badge">Expert</span> Performance</summary>
      <p class="sub">
        Each setting trades something for speed; measured on this laptop in <code>docs/PERFORMANCE.md</code>.
        Saving restarts the model.
      </p>
      {#if speculative}
        {@render setting(PERF_SETTINGS.ngram, ngramControl)}
        {@render setting(PERF_SETTINGS.mtp, mtpControl, model.has_mtp ? null : "This model file has no MTP layers.")}
      {/if}
      {#if device === "gpu"}
        {@render setting(PERF_SETTINGS.gpu_api, gpuControl)}
      {/if}
    </details>
  {/if}

  {#snippet ngramControl()}<input type="checkbox" bind:checked={ngram} />{/snippet}
  {#snippet mtpControl()}<input type="checkbox" bind:checked={mtp} disabled={!model.has_mtp && !mtp} />{/snippet}
  {#snippet gpuControl()}
    <select bind:value={gpuApi}>
      <option value="default">Default{backends ? ` (${API_NAME[backends.settings.gpu_api]})` : ""}</option>
      <option value="vulkan">Vulkan</option>
      <option value="rocm" disabled={backends !== null && !hasBuild("rocm") && initial.gpu_api !== "rocm"}>
        ROCm{backends && !hasBuild("rocm") ? " (not installed)" : ""}
      </option>
    </select>
  {/snippet}
  {#snippet setting(s: (typeof PERF_SETTINGS)[keyof typeof PERF_SETTINGS], control: import("svelte").Snippet, blocked: string | null = null)}
    <div class="perf">
      <label class="perf-head">{@render control()} <b>{s.title}</b></label>
      <p class="sub">{s.what}{blocked ? ` ${blocked}` : ""}</p>
      <ul>
        {#each s.upsides as u (u)}<li><span class="mark" aria-label="upside">+</span>{u}</li>{/each}
        {#each s.downsides as d (d)}<li><span class="mark" aria-label="downside">−</span>{d}</li>{/each}
      </ul>
    </div>
  {/snippet}

  <div class="row aliases">
    <span class="label">Aliases</span>
    {#each mine as a (a.name)}
      <span class="tag" title={a.managed ? "" : "from config.toml"}>
        {a.name}
        {#if a.managed}
          <button class="x" aria-label="Remove alias {a.name}" onclick={() => aliasAction(() => deleteAlias(a.name))}>✕</button>
        {/if}
      </span>
    {/each}
    <input
      class="alias-input"
      placeholder="new alias, e.g. coder"
      bind:value={newAlias}
      onkeydown={(e) => e.key === "Enter" && newAlias.trim() && aliasAction(() => setAlias(newAlias.trim(), model.id))}
    />
    <button class="small" disabled={!newAlias.trim()} onclick={() => aliasAction(() => setAlias(newAlias.trim(), model.id))}>
      Add
    </button>
  </div>
  {#if error}<p class="error-text">{error}</p>{/if}
  <div class="row">
    <span class="sub">
      {model.state === "ready" || model.state === "loading" ? "Saving restarts the model on its next request." : ""}
    </span>
    <span class="spacer"></span>
    {#if model.overridden}
      <button disabled={saving} onclick={reset} title="Drop the overrides and use the settings from config.toml">Reset to config.toml</button>
    {/if}
    <button onclick={onclose}>Cancel</button>
    <button class="primary" disabled={!dirty || saving} onclick={save}>{saving ? "Saving…" : "Save"}</button>
  </div>
</div>

<style>
  .editor {
    display: grid;
    gap: 0.6rem;
    padding: 0.4rem 0 0.6rem;
  }
  .file {
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .grid {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 1.5rem;
  }
  .grid label {
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.9rem;
    color: var(--muted);
  }
  .grid input[type="number"] {
    width: 6.5rem;
  }
  .wide {
    flex-basis: 100%;
  }
  .wide input {
    flex: 1;
  }
  .mono {
    font-family: ui-monospace, monospace;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .spacer {
    flex: 1;
  }
  .label {
    color: var(--muted);
    font-size: 0.9rem;
    margin-right: 0.2rem;
  }
  .x {
    border: none;
    background: none;
    padding: 0 0 0 0.2rem;
    font-size: 0.7rem;
    cursor: pointer;
    color: var(--muted);
  }
  .alias-input {
    width: 11rem;
  }
  p {
    margin: 0;
  }
  .expert {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
  }
  .expert summary {
    cursor: pointer;
    font-weight: 600;
  }
  .expert > p {
    margin: 0.4rem 0 0.2rem;
  }
  .badge {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid var(--warn, var(--border));
    color: var(--warn, var(--muted));
    border-radius: 4px;
    padding: 0 0.3rem;
    margin-right: 0.3rem;
  }
  .perf {
    padding: 0.5rem 0;
    border-top: 1px solid var(--border);
    margin-top: 0.4rem;
  }
  .perf-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .perf p {
    margin: 0.2rem 0;
  }
  .perf ul {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: 0.85rem;
  }
  .mark {
    display: inline-block;
    width: 1.1rem;
    font-weight: 700;
    color: var(--muted);
  }
</style>
