<script lang="ts">
  import { onMount } from "svelte";
  import {
    DAEMON_URL,
    fetchHardware,
    formatBytes,
    profileLabel,
    subscribeMetrics,
    type ConnectionState,
  } from "$lib/daemon";
  import type { HardwareInfo, MemoryPool, Metrics } from "$lib/types";

  let hw = $state<HardwareInfo | null>(null);
  let hwError = $state<string | null>(null);
  let metrics = $state<Metrics | null>(null);
  let conn = $state<ConnectionState>("connecting");

  // Live pools win over the startup snapshot.
  let pools = $derived<MemoryPool[]>(metrics?.pools ?? hw?.pools ?? []);
  let power = $derived(metrics?.power ?? hw?.power ?? null);

  async function loadHardware() {
    hwError = null;
    try {
      hw = await fetchHardware();
    } catch (e) {
      hwError = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(() => {
    loadHardware();
    return subscribeMetrics(
      (m) => (metrics = m),
      (s) => {
        conn = s;
        if (s === "open" && !hw) loadHardware();
      },
    );
  });

  const poolLabel = { gpu: "GPU pool (VRAM)", system: "System pool (CPU / NPU)" } as const;
  const pct = (p: MemoryPool) => (p.total_bytes ? (100 * p.used_bytes) / p.total_bytes : 0);
</script>

<main>
  <header>
    <h1>contio</h1>
    <span class="conn {conn}" title={DAEMON_URL}>
      {conn === "open" ? "connected" : conn === "connecting" ? "connecting…" : "daemon offline"}
    </span>
  </header>

  {#if hwError && !hw}
    <section class="card error">
      <p>Can't reach <code>contiod</code> at <code>{DAEMON_URL}</code>: {hwError}</p>
      <p>Start it with <code>cargo run -p contiod</code>.</p>
      <button onclick={loadHardware}>Retry</button>
    </section>
  {/if}

  {#if hw}
    <section class="card">
      <h2>Memory</h2>
      {#each pools as pool (pool.kind)}
        <div class="pool">
          <div class="pool-head">
            <span>{poolLabel[pool.kind]}</span>
            <span class="num">{formatBytes(pool.used_bytes)} / {formatBytes(pool.total_bytes)}</span>
          </div>
          <div class="bar" role="meter" aria-valuenow={pct(pool)} aria-valuemin={0} aria-valuemax={100}
               aria-label={poolLabel[pool.kind]}>
            <div class="fill" style:width="{pct(pool)}%"></div>
          </div>
        </div>
      {/each}
    </section>

    <section class="stats">
      <div class="card stat">
        <span class="label">SoC power</span>
        <span class="value num">{metrics?.soc_watts != null ? `${metrics.soc_watts.toFixed(1)} W` : "–"}</span>
      </div>
      <div class="card stat">
        <span class="label">GPU busy</span>
        <span class="value num">{metrics?.gpu_busy_percent != null ? `${metrics.gpu_busy_percent} %` : "–"}</span>
      </div>
      <div class="card stat">
        <span class="label">Power profile</span>
        <span class="value">{power ? profileLabel(power.profile) : "–"}</span>
        <span class="sub">
          {power?.on_ac ? "on AC" : power?.on_ac === false ? "on battery" : ""}
          {power?.battery_percent != null ? `· ${power.battery_percent} %` : ""}
        </span>
      </div>
    </section>

    <section class="card">
      <h2>Devices</h2>
      <dl>
        <dt>CPU</dt>
        <dd>
          {hw.cpu.model}<br />
          <span class="sub">
            {hw.cpu.cores} cores / {hw.cpu.threads} threads ·
            {hw.cpu.core_classes.map((c) => `${c.cores}× ${(c.max_mhz / 1000).toFixed(1)} GHz`).join(" + ")}
            {hw.cpu.avx512 ? "· AVX-512" : ""}
          </span>
        </dd>
        <dt>GPU</dt>
        <dd>
          {#if hw.gpu}
            {hw.gpu.driver} <code>{hw.gpu.pci_id}</code><br />
            <span class="sub">
              {formatBytes(hw.gpu.vram_total_bytes)} VRAM + {formatBytes(hw.gpu.gtt_total_bytes)} GTT
            </span>
          {:else}
            <span class="sub">not detected</span>
          {/if}
        </dd>
        <dt>NPU</dt>
        <dd>
          {#if hw.npu}
            {hw.npu.driver} <code>{hw.npu.device_path}</code>
            {#if !hw.npu.accessible}<span class="warn">no access (check the <code>render</code> group)</span>{/if}
          {:else}
            <span class="sub">not detected</span>
          {/if}
        </dd>
      </dl>
    </section>

    <section class="card">
      <h2>API</h2>
      <p>OpenAI-compatible endpoint: <code>{DAEMON_URL}/v1</code></p>
      <p class="sub">No models installed yet.</p>
    </section>
  {/if}
</main>

<style>
  :global(:root) {
    --bg: #f6f7f9;
    --card: #ffffff;
    --text: #1d2129;
    --muted: #677084;
    --border: #e1e4ea;
    --accent: #2f6fde;
    --ok: #1f8a4c;
    --warn: #b7791f;
    --err: #c53030;
    font-family: system-ui, sans-serif;
    font-size: 15px;
    color: var(--text);
    background: var(--bg);
  }
  @media (prefers-color-scheme: dark) {
    :global(:root) {
      --bg: #15171c;
      --card: #1e2128;
      --text: #e6e8ec;
      --muted: #9aa3b2;
      --border: #2e323b;
      --accent: #6ea0ff;
    }
  }
  :global(body) {
    margin: 0;
  }
  main {
    max-width: 760px;
    margin: 0 auto;
    padding: 1.25rem;
    display: grid;
    gap: 1rem;
  }
  header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  h1 {
    margin: 0;
    font-size: 1.5rem;
  }
  h2 {
    margin: 0 0 0.75rem;
    font-size: 1rem;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
  }
  .conn {
    font-size: 0.85rem;
    color: var(--muted);
  }
  .conn::before {
    content: "●";
    margin-right: 0.35rem;
  }
  .conn.open::before {
    color: var(--ok);
  }
  .conn.connecting::before {
    color: var(--warn);
  }
  .conn.closed::before {
    color: var(--err);
  }
  .error {
    border-color: var(--err);
  }
  .pool + .pool {
    margin-top: 0.75rem;
  }
  .pool-head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
  }
  .bar {
    height: 10px;
    border-radius: 5px;
    background: var(--border);
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--accent);
    transition: width 0.4s ease;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .label,
  .sub {
    color: var(--muted);
    font-size: 0.85rem;
  }
  .value {
    font-size: 1.4rem;
    font-weight: 600;
  }
  .num {
    font-variant-numeric: tabular-nums;
  }
  .warn {
    color: var(--warn);
    margin-left: 0.5rem;
  }
  dl {
    display: grid;
    grid-template-columns: 4rem 1fr;
    gap: 0.6rem;
    margin: 0;
  }
  dt {
    color: var(--muted);
  }
  dd {
    margin: 0;
  }
  code {
    font-size: 0.85em;
  }
  button {
    font: inherit;
    padding: 0.35rem 0.9rem;
  }
</style>
