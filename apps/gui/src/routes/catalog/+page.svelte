<script lang="ts">
  import { page } from "$app/state";
  import { errorText, fetchCatalog, formatBytes, formatParams, searchHf } from "$lib/daemon";
  import { live } from "$lib/live.svelte";
  import FitPill from "$lib/components/FitPill.svelte";
  import ModelDetail from "$lib/components/ModelDetail.svelte";
  import type { CatalogOverview, ModelKind, ModelSummary, RepoInfo } from "$lib/types";

  let overview = $state<CatalogOverview | null>(null);
  let error = $state<string | null>(null);
  let refreshing = $state(false);

  let query = $state("");
  let device = $state<"any" | "gpu" | "cpu" | "npu">("any");
  let kind = $state<"all" | ModelKind>("all");
  let fitsOnly = $state(false);
  let caps = $state<string[]>([]);
  let hf = $state<RepoInfo[] | null>(null);
  let hfBusy = $state(false);
  // `/catalog?model=<id>` or `?repo=<owner/name>` opens the detail dialog
  const linked = page.url.searchParams;
  let selected = $state<{ id: string } | { repo: string } | null>(
    linked.get("model") ? { id: linked.get("model")! } : linked.get("repo") ? { repo: linked.get("repo")! } : null,
  );

  const CAPS = ["tools", "reasoning", "vision", "code"];

  async function load(refresh = false) {
    refreshing = true;
    error = null;
    try {
      overview = await fetchCatalog(refresh);
    } catch (e) {
      error = errorText(e);
    } finally {
      refreshing = false;
    }
  }

  $effect(() => {
    if (live.online && !overview) load();
  });

  function keep(m: ModelSummary): boolean {
    const hay = `${m.id} ${m.name} ${m.family}`.toLowerCase();
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.every((w) => hay.includes(w))) return false;
    if (kind !== "all" && m.kind !== kind) return false;
    if (!caps.every((c) => m.capabilities.includes(c))) return false;
    if (device === "npu") return m.npu.length > 0;
    const fit = device === "cpu" ? m.suggested?.fit.cpu : m.suggested?.fit.gpu;
    if (device !== "any" && !m.suggested) return false;
    if (fitsOnly && (!fit || fit.verdict === "too_big")) return false;
    return true;
  }

  let groups = $derived(
    (overview?.families ?? [])
      .map((f) => ({ family: f, models: (overview?.models ?? []).filter((m) => m.family === f.id && keep(m)) }))
      .filter((g) => g.models.length > 0),
  );
  let shown = $derived(groups.reduce((n, g) => n + g.models.length, 0));

  async function searchHub() {
    if (!query.trim()) return;
    hfBusy = true;
    try {
      hf = await searchHf(query.trim());
    } catch (e) {
      error = errorText(e);
    } finally {
      hfBusy = false;
    }
  }

  function toggleCap(c: string) {
    caps = caps.includes(c) ? caps.filter((x) => x !== c) : [...caps, c];
  }

  const sizeRange = (m: ModelSummary) =>
    m.min_size == null ? "" : m.min_size === m.max_size ? formatBytes(m.min_size) : `${formatBytes(m.min_size)} – ${formatBytes(m.max_size ?? 0)}`;
</script>

<section class="card filters">
  <input
    class="search"
    type="search"
    placeholder="Search models (e.g. qwen coder, gemma, embed)…"
    bind:value={query}
    onkeydown={(e) => e.key === "Enter" && hf !== null && searchHub()}
  />
  <div class="row">
    <span class="label">Device</span>
    {#each ["any", "gpu", "cpu", "npu"] as const as d (d)}
      <button class="small" class:primary={device === d} onclick={() => (device = d)}>{d.toUpperCase()}</button>
    {/each}
    <span class="label">Type</span>
    <select bind:value={kind}>
      <option value="all">all</option>
      <option value="chat">chat</option>
      <option value="embedding">embedding</option>
      <option value="reranker">reranker</option>
    </select>
    <span class="label">Needs</span>
    {#each CAPS as c (c)}
      <button class="small" class:primary={caps.includes(c)} onclick={() => toggleCap(c)}>{c}</button>
    {/each}
    <label class="fits"><input type="checkbox" bind:checked={fitsOnly} /> fits my machine</label>
  </div>
  <div class="row sub">
    {#if overview}{shown} of {overview.models.length} models{/if}
    {#if overview && !overview.npu_available}· NPU models appear once FastFlowLM is set up{/if}
    <span class="spacer"></span>
    <button class="small" disabled={!query.trim() || hfBusy} onclick={searchHub}>
      {hfBusy ? "Searching…" : "Search Hugging Face"}
    </button>
    <button class="small" disabled={refreshing} onclick={() => load(true)}>{refreshing ? "Refreshing…" : "Refresh"}</button>
  </div>
</section>

{#if error}<p class="error-text">{error}</p>{/if}

{#if hf !== null}
  <section class="card">
    <div class="row head">
      <h2>Hugging Face: “{query}”</h2>
      <span class="sub">community uploads, not curated</span>
      <span class="spacer"></span>
      <button class="small" onclick={() => (hf = null)}>Close</button>
    </div>
    {#if hf.length === 0}
      <p class="sub">No GGUF repos found.</p>
    {:else}
      <table class="list">
        <thead><tr><th>Repo</th><th>Params</th><th>Arch</th><th>Downloads</th><th></th></tr></thead>
        <tbody>
          {#each hf as h (h.repo)}
            <tr class="clickable" onclick={() => (selected = { repo: h.repo })}>
              <td>{h.repo}</td>
              <td class="num">{formatParams(h.params)}</td>
              <td>{h.arch ?? "–"}</td>
              <td class="num">{h.downloads.toLocaleString()}</td>
              <td>
                {#if h.tools}<span class="tag">tools</span>{/if}
                {#if h.reasoning}<span class="tag">reasoning</span>{/if}
                {#if h.gated}<span class="tag">gated</span>{/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
{/if}

{#if !overview && !error}
  <p class="sub">{live.online ? "Loading catalog…" : "Waiting for contiod…"}</p>
{/if}

{#each groups as g (g.family.id)}
  <section class="card">
    <div class="row head">
      <h2>{g.family.name}</h2>
      <span class="sub">{g.family.vendor} · {g.family.license}</span>
    </div>
    <p class="sub summary">{g.family.summary}</p>
    <table class="list fixed">
      <colgroup>
        <col /><col style="width: 9rem" /><col style="width: 11rem" /><col style="width: 5.5rem" />
        <col style="width: 5.5rem" /><col style="width: 11rem" /><col style="width: 1.5rem" />
      </colgroup>
      <thead>
        <tr><th>Model</th><th>Params</th><th>Suggested</th><th>GPU</th><th>CPU</th><th>Quants · sizes</th><th></th></tr>
      </thead>
      <tbody>
        {#each g.models as m (m.id)}
          <tr class="clickable" onclick={() => (selected = { id: m.id })}>
            <td>
              <b>{m.name}</b>
              {#if m.installed.length}<span class="tag accent">installed</span>{/if}
              <div>
                {#if m.kind !== "chat"}<span class="tag">{m.kind}</span>{/if}
                {#each m.capabilities as c (c)}<span class="tag">{c}</span>{/each}
                {#if m.npu.length}<span class="tag">npu</span>{/if}
              </div>
            </td>
            <td class="num">{formatParams(m.params, m.active_b)}</td>
            <td>
              {#if m.suggested}{m.suggested.label} <span class="sub">{formatBytes(m.suggested.size)}</span>
              {:else if m.error}<span class="sub" title={m.error}>unavailable</span>
              {:else}<span class="sub">NPU only</span>{/if}
            </td>
            <td><FitPill fit={m.suggested?.fit.gpu ?? null} /></td>
            <td><FitPill fit={m.suggested?.fit.cpu ?? null} /></td>
            <td class="sub num">{m.quant_count ? `${m.quant_count} · ${sizeRange(m)}` : ""}</td>
            <td class="go">›</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </section>
{/each}

{#if overview && shown === 0}
  <p class="sub">No catalog model matches. Try <b>Search Hugging Face</b> for community uploads.</p>
{/if}

{#if selected}
  <ModelDetail target={selected} onclose={() => (selected = null)} />
{/if}

<style>
  .filters {
    display: grid;
    gap: 0.6rem;
  }
  .search {
    width: 100%;
    box-sizing: border-box;
    font-size: 1rem;
    padding: 0.5rem 0.7rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .row.head {
    gap: 0.75rem;
  }
  .row.head h2 {
    margin: 0;
  }
  .label {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0 0.1rem 0 0.6rem;
  }
  .label:first-child {
    margin-left: 0;
  }
  .fits {
    margin-left: 0.6rem;
    font-size: 0.9rem;
  }
  .spacer {
    flex: 1;
  }
  .summary {
    margin: 0.25rem 0 0.6rem;
  }
  table.fixed {
    table-layout: fixed;
  }
  tr.clickable {
    cursor: pointer;
  }
  tr.clickable:hover td {
    background: var(--bg);
  }
  .go {
    color: var(--muted);
    text-align: right;
    font-size: 1.2rem;
  }
</style>
