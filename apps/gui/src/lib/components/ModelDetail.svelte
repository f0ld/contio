<script lang="ts">
  import {
    errorText,
    fetchCatalogModel,
    fetchRepo,
    formatBytes,
    formatContext,
    formatParams,
    installModel,
  } from "$lib/daemon";
  import { jobs, refreshJobs } from "$lib/jobs.svelte";
  import FitPill from "./FitPill.svelte";
  import type { DeviceKind, Fit, ModelDetail, QuantView } from "$lib/types";

  /** A catalog id (`qwen3-8b`, `flm:<tag>`) or a Hugging Face repo. */
  let { target, onclose }: { target: { id: string } | { repo: string }; onclose: () => void } = $props();

  let detail = $state<ModelDetail | null>(null);
  let loadError = $state<string | null>(null);
  let device = $state<DeviceKind>("gpu");
  let quant = $state<string | null>(null);
  /** The user picked a quant; don't re-suggest when device or context change. */
  let quantPicked = $state(false);
  let refitting = $state(false);
  let npuTag = $state<string | null>(null);
  let context = $state<number | null>(null);
  let name = $state("");
  let run = $state(true);
  let jobId = $state<number | null>(null);
  let installError = $state<string | null>(null);

  let job = $derived(jobs.byId(jobId));
  let devices = $derived.by(() => {
    const d: DeviceKind[] = [];
    if (detail?.quants.length) d.push("gpu", "cpu");
    if (detail?.npu.some((n) => !n.unsupported)) d.push("npu");
    return d;
  });

  const fitOf = (q: QuantView, d: DeviceKind): Fit | null => (d === "cpu" ? q.fit.cpu : q.fit.gpu);

  /** Same rule as the daemon: recommended if it fits, else the largest comfortable fit. */
  function suggest(d: ModelDetail, dev: DeviceKind): string | null {
    const ok = (q: QuantView) => fitOf(q, dev)?.verdict !== "too_big";
    const rec = d.quants.find((q) => q.recommended);
    if (rec && ok(rec)) return rec.label;
    const rev = [...d.quants].reverse();
    return (rev.find((q) => fitOf(q, dev)?.verdict === "fits") ?? rev.find(ok) ?? d.quants[0])?.label ?? null;
  }

  const fetchDetail = (t: typeof target, ctx?: number) => ("id" in t ? fetchCatalogModel(t.id, ctx) : fetchRepo(t.repo, ctx));

  $effect(() => {
    const t = target;
    detail = null;
    loadError = null;
    jobId = null;
    quantPicked = false;
    fetchDetail(t)
      .then((d) => {
        detail = d;
        device = d.quants.length ? "gpu" : "npu";
        quant = suggest(d, "gpu");
        npuTag = d.npu.find((n) => !n.unsupported)?.tag ?? null;
        context = d.default_context;
      })
      .catch((e) => (loadError = errorText(e)));
  });

  // recompute the fits for a changed context (debounced while typing)
  $effect(() => {
    const ctx = context;
    const d = detail;
    if (!d || !d.quants.length || !ctx || ctx < 512 || ctx === d.fit_context) return;
    const t = target;
    const timer = setTimeout(async () => {
      refitting = true;
      try {
        const next = await fetchDetail(t, ctx);
        if (target !== t || context !== ctx) return;
        detail = next;
        if (!quantPicked && device !== "npu") quant = suggest(next, device);
      } catch (e) {
        loadError = errorText(e);
      } finally {
        refitting = false;
      }
    }, 350);
    return () => clearTimeout(timer);
  });

  function pickDevice(d: DeviceKind) {
    device = d;
    if (detail && d !== "npu" && !quantPicked) quant = suggest(detail, d);
  }

  async function install() {
    if (!detail) return;
    installError = null;
    const base = detail.id && !detail.id.startsWith("flm:") ? { model: detail.id } : {};
    const req =
      device === "npu"
        ? { ...base, flm: npuTag ?? undefined, device }
        : { ...(detail.id ? base : { repo: detail.repo ?? undefined }), quant: quant ?? undefined, device };
    try {
      const r = await installModel({
        ...req,
        context: context && context !== detail.default_context ? context : undefined,
        id: name.trim() || undefined,
        load: run,
      });
      jobId = r.job_id;
      refreshJobs();
    } catch (e) {
      installError = errorText(e);
    }
  }

  const onkey = (e: KeyboardEvent) => e.key === "Escape" && onclose();
  let progress = $derived(job?.progress?.total ? (100 * job.progress.done) / job.progress.total : null);
</script>

<svelte:window onkeydown={onkey} />

<div class="backdrop" role="presentation" onclick={(e) => e.target === e.currentTarget && onclose()}>
  <div class="panel card" role="dialog" aria-modal="true" aria-label={detail?.name ?? "Model"}>
    <button class="close" onclick={onclose} aria-label="Close">✕</button>
    {#if loadError}
      <p class="error-text">{loadError}</p>
    {:else if !detail}
      <p class="sub">Loading…</p>
    {:else}
      <h2>{detail.name}</h2>
      <p class="sub">
        {#if detail.repo}<a href="https://huggingface.co/{detail.repo}" target="_blank" rel="noreferrer">{detail.repo}</a> ·{/if}
        {formatParams(detail.params, detail.active_b)} · {detail.kind === "transcription" ? "speech to text" : detail.kind}
        {#if detail.kind !== "transcription"}· up to {formatContext(detail.context_length)} context{/if}
      </p>
      <p>
        {#each detail.capabilities as c (c)}<span class="tag accent">{c}</span>{/each}
      </p>
      {#if detail.notes}<p class="sub">{detail.notes}</p>{/if}
      {#if detail.error}<p class="error-text">Couldn't list files: {detail.error}</p>{/if}

      <div class="row">
        <span class="label">Run on</span>
        {#each devices as d (d)}
          <button class:primary={device === d} onclick={() => pickDevice(d)}>{d.toUpperCase()}</button>
        {/each}
        {#if devices.length === 0}<span class="sub">nothing installable</span>{/if}
      </div>

      {#if device === "npu"}
        <table class="list">
          <thead><tr><th></th><th>FastFlowLM model</th><th>Memory</th><th>Context</th><th>Fit</th><th></th></tr></thead>
          <tbody>
            {#each detail.npu as n (n.tag)}
              <tr class:selected={npuTag === n.tag}>
                <td><input type="radio" name="npu" value={n.tag} bind:group={npuTag} disabled={n.unsupported} /></td>
                <td>
                  <code>{n.tag}</code>
                  {#if n.single_turn}<span class="tag">single-turn</span>{/if}
                  {#if n.unsupported}<span class="tag" title="Set asr = true under [backends.flm] in config.toml; FastFlowLM serves Whisper next to an NPU LLM">needs asr = true</span>{/if}
                </td>
                <td class="num">{n.footprint_bytes ? formatBytes(n.footprint_bytes) : "–"}</td>
                <td class="num">{formatContext(n.default_context)}</td>
                <td><FitPill fit={n.fit} /></td>
                <td class="sub">{n.installed.length ? `installed: ${n.installed.join(", ")}` : n.downloaded ? "downloaded" : ""}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else if detail.quants.length}
        <table class="list">
          <thead>
            <tr><th></th><th>Quant</th><th>Size</th><th>Bits/w</th><th>Quality</th><th>Memory needed</th><th></th></tr>
          </thead>
          <tbody>
            {#each detail.quants as q (q.label)}
              <tr class:selected={quant === q.label}>
                <td><input type="radio" name="quant" value={q.label} bind:group={quant} onchange={() => (quantPicked = true)} /></td>
                <td>{q.label}{#if q.recommended}<span class="tag accent">recommended</span>{/if}</td>
                <td class="num">{formatBytes(q.size)}{q.files > 1 ? ` (${q.files} files)` : ""}</td>
                <td class="num">{q.bits_per_weight?.toFixed(2) ?? "–"}</td>
                <td>{q.tier ?? "–"}</td>
                <td><FitPill fit={fitOf(q, device)} detail /></td>
                <td class="sub">{q.installed.length ? `installed: ${q.installed.join(", ")}` : q.downloaded ? "downloaded" : ""}</td>
              </tr>
            {/each}
          </tbody>
        </table>
        <p class="sub">
          {#if detail.kind === "transcription"}Memory for 30-second audio windows.{:else}
          Memory at {formatContext(detail.fit_context)} context{refitting ? " (updating…)" : ""}{detail.mmproj ? `, vision projector included (${formatBytes(detail.mmproj.size)})` : ""}.{/if}
          Fit compares against {device === "cpu" ? "system" : "GPU"} memory minus the reserve.
        </p>
      {/if}

      {#if devices.length}
        <div class="row options">
          {#if detail.kind !== "transcription"}
            <label>
              Context <input type="number" min="512" max={detail.context_length ?? undefined} step="1024" bind:value={context} />
              {#if detail.context_length}<span class="sub">max {formatContext(detail.context_length)}</span>{/if}
            </label>
          {/if}
          <label>API name <input placeholder="automatic" bind:value={name} /></label>
          <label><input type="checkbox" bind:checked={run} /> start when done</label>
          <button class="primary" disabled={!!job && job.state === "running"} onclick={install}>
            {run ? "Install & run" : "Install"}
          </button>
        </div>
      {/if}
      {#if installError}<p class="error-text">{installError}</p>{/if}

      {#if job}
        <div class="job">
          <div class="row">
            <b>Job #{job.id}</b>
            <span class="sub">{job.state === "running" ? job.step ?? "starting" : job.state}</span>
            <a href="/jobs">all jobs</a>
          </div>
          {#if job.state === "running" && progress != null}
            <div class="progress"><div style:width="{progress}%"></div></div>
          {/if}
          {#if job.state === "done"}
            <p>
              Installed as <code>{String(job.result?.model ?? "")}</code>. Use that name as <code>model</code> in the
              API{#if job.result?.kind === "chat" || job.result?.kind === "transcription"}, <a href="/playground?model={encodeURIComponent(String(job.result?.model ?? ""))}">try it in the
                playground</a>{/if}, or manage it on the <a href="/">dashboard</a>.
            </p>
          {/if}
          {#if job.error}<p class="error-text">{job.error}</p>{/if}
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 20;
    background: rgb(0 0 0 / 0.35);
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 3rem 1rem;
    overflow: auto;
  }
  .panel {
    position: relative;
    width: min(900px, 100%);
  }
  .close {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    border: none;
  }
  h2 {
    font-size: 1.2rem;
  }
  p {
    margin: 0.4rem 0;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.9rem 0 0.6rem;
  }
  .label {
    color: var(--muted);
    margin-right: 0.25rem;
  }
  .options label {
    color: var(--muted);
    font-size: 0.9rem;
  }
  .options input[type="number"] {
    width: 7rem;
  }
  .options button {
    margin-left: auto;
  }
  tr.selected td {
    background: var(--bg);
  }
  .job {
    margin-top: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--border);
  }
  .job .row {
    margin: 0.3rem 0;
  }
</style>
