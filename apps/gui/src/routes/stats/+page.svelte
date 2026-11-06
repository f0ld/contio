<script lang="ts">
  import { onMount } from "svelte";
  import StackedColumns, { type Column } from "$lib/components/StackedColumns.svelte";
  import TimeSeries from "$lib/components/TimeSeries.svelte";
  import { compactNumber, errorText, fetchDailyUsage, fetchLiveUsage, fetchUsage } from "$lib/daemon";
  import { live as conn } from "$lib/live.svelte";
  import type { DailyUsage, LiveUsage, UsageReport } from "$lib/types";

  let liveData = $state<LiveUsage | null>(null);
  let daily = $state<DailyUsage | null>(null);
  let usage = $state<UsageReport | null>(null);
  let error = $state<string | null>(null);

  let range = $state(30);
  let measure = $state<"tokens" | "generated" | "prompt" | "requests">("tokens");
  /** seconds of moving average on the live curve (1 = raw) */
  let smooth = $state(5);

  async function pollLive() {
    try {
      liveData = await fetchLiveUsage();
      error = null;
    } catch (e) {
      error = errorText(e);
    }
  }

  async function loadDaily() {
    try {
      [daily, usage] = await Promise.all([fetchDailyUsage(range), fetchUsage()]);
    } catch (e) {
      error = errorText(e);
    }
  }

  onMount(() => {
    pollLive();
    loadDaily();
    const a = setInterval(pollLive, 1000);
    const b = setInterval(loadDaily, 30_000);
    return () => {
      clearInterval(a);
      clearInterval(b);
    };
  });

  $effect(() => {
    void range;
    void conn.online;
    loadDaily();
  });

  // --- live ---
  let points = $derived.by(() => {
    const s = liveData?.series ?? [];
    // drop the second that is still filling, average over `smooth` seconds
    const done = s.slice(0, -1);
    return done.map((p, i) => {
      const from = Math.max(0, i - smooth + 1);
      const win = done.slice(from, i + 1);
      return { t: p.t, v: win.reduce((a, x) => a + x.generated, 0) / win.length };
    });
  });
  let nowTps = $derived(liveData?.models.reduce((a, m) => a + m.generated_tps_10s, 0) ?? 0);
  let activeModels = $derived((liveData?.models ?? []).filter((m) => m.in_flight > 0 || m.requests_60s > 0 || m.generated_tps_60s > 0));
  let today = $derived(
    (usage?.models ?? []).reduce(
      (a, m) => ({ tokens: a.tokens + m.today.prompt_tokens + m.today.completion_tokens, requests: a.requests + m.today.requests }),
      { tokens: 0, requests: 0 },
    ),
  );

  // --- daily ---
  const value = (r: DailyUsage["rows"][number]) =>
    measure === "tokens"
      ? r.prompt_tokens + r.completion_tokens
      : measure === "generated"
        ? r.completion_tokens
        : measure === "prompt"
          ? r.prompt_tokens
          : r.requests;
  /** at most 7 named series with values for the chosen measure; the rest fold into "other" */
  let series = $derived.by(() => {
    const has = new Set((daily?.rows ?? []).filter((r) => value(r) > 0).map((r) => r.model));
    const m = (daily?.models ?? []).filter((x) => has.has(x));
    return m.length > 7 ? [...m.slice(0, 7), "other"] : m;
  });
  let columns = $derived.by((): Column[] => {
    if (!daily) return [];
    const named = new Set(series);
    return daily.days.map((day) => {
      const values: Record<string, number> = {};
      for (const r of daily!.rows.filter((r) => r.day === day && value(r) > 0)) {
        const key = named.has(r.model) ? r.model : "other";
        values[key] = (values[key] ?? 0) + value(r);
      }
      const d = new Date(`${day}T12:00:00`);
      return { key: day, label: d.toLocaleDateString(undefined, { month: "short", day: "numeric" }), values };
    });
  });
  let rangeTotals = $derived.by(() => {
    const t = new Map<string, { requests: number; prompt: number; generated: number; audio: number }>();
    for (const r of daily?.rows ?? []) {
      const x = t.get(r.model) ?? { requests: 0, prompt: 0, generated: 0, audio: 0 };
      x.requests += r.requests;
      x.prompt += r.prompt_tokens;
      x.generated += r.completion_tokens;
      x.audio += r.audio_seconds;
      t.set(r.model, x);
    }
    return [...t.entries()].sort((a, b) => b[1].prompt + b[1].generated - (a[1].prompt + a[1].generated));
  });
  let dayRows = $derived(
    [...(daily?.days ?? [])].reverse().map((day) => {
      const rows = daily!.rows.filter((r) => r.day === day);
      return {
        day,
        requests: rows.reduce((a, r) => a + r.requests, 0),
        prompt: rows.reduce((a, r) => a + r.prompt_tokens, 0),
        generated: rows.reduce((a, r) => a + r.completion_tokens, 0),
      };
    }).filter((r) => r.requests > 0),
  );
  const fmt = (v: number) => (measure === "requests" ? `${v} requests` : `${compactNumber(v)} tokens`);
</script>

{#if error}<p class="error-text">{error}</p>{/if}

<section class="card">
  <div class="head">
    <h2>Live</h2>
    <span class="sub">generated tokens per second, all models, last 5 minutes</span>
    <span class="spacer"></span>
    <label class="sub">smoothing
      <select bind:value={smooth}>
        <option value={1}>off</option>
        <option value={5}>5 s</option>
        <option value={15}>15 s</option>
      </select>
    </label>
  </div>
  <div class="tiles">
    <div class="tile"><span class="label">Generating now</span><span class="value">{nowTps.toFixed(1)}</span><span class="sub">tokens / s (10 s)</span></div>
    <div class="tile"><span class="label">Requests running</span><span class="value">{liveData?.in_flight ?? 0}</span><span class="sub">across all models</span></div>
    <div class="tile"><span class="label">Tokens today</span><span class="value">{compactNumber(today.tokens)}</span><span class="sub">prompt + generated</span></div>
    <div class="tile"><span class="label">Requests today</span><span class="value">{compactNumber(today.requests)}</span><span class="sub">since midnight</span></div>
  </div>
  <TimeSeries {points} unit="tok/s" format={(v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))} />
  {#if activeModels.length}
    <table class="list">
      <thead>
        <tr><th>Model</th><th class="num">Now (10 s)</th><th class="num">Last minute</th><th class="num">Prompt tok/s</th><th class="num">Requests / min</th><th class="num">Running</th></tr>
      </thead>
      <tbody>
        {#each activeModels as m (m.model)}
          <tr>
            <td><b>{m.model}</b></td>
            <td class="num">{m.generated_tps_10s.toFixed(1)} tok/s</td>
            <td class="num">{m.generated_tps_60s.toFixed(1)} tok/s</td>
            <td class="num">{m.prompt_tps_60s.toFixed(0)}</td>
            <td class="num">{m.requests_60s}</td>
            <td class="num">{m.in_flight}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {:else}
    <p class="sub">Nothing served in the last minute. Throughput appears here while models answer requests.</p>
  {/if}
</section>

<section class="card">
  <div class="head">
    <h2>Daily usage</h2>
    <span class="spacer"></span>
    <label class="sub">show
      <select bind:value={measure}>
        <option value="tokens">all tokens</option>
        <option value="generated">generated tokens</option>
        <option value="prompt">prompt tokens</option>
        <option value="requests">requests</option>
      </select>
    </label>
    {#each [7, 30, 90] as d (d)}
      <button class="small" class:primary={range === d} onclick={() => (range = d)}>{d} days</button>
    {/each}
  </div>
  {#if daily && daily.rows.length}
    <StackedColumns {columns} {series} format={fmt} />
    <div class="tables">
      <div>
        <h3>By model, last {range} days</h3>
        <table class="list">
          <thead><tr><th>Model</th><th class="num">Requests</th><th class="num">Prompt</th><th class="num">Generated</th></tr></thead>
          <tbody>
            {#each rangeTotals as [model, t] (model)}
              <tr>
                <td>{model}</td>
                <td class="num">{t.requests.toLocaleString()}</td>
                <td class="num">{compactNumber(t.prompt)}</td>
                <td class="num">{t.audio && !t.generated ? `${Math.round(t.audio)} s audio` : compactNumber(t.generated)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <div class="scroll">
        <h3>By day</h3>
        <table class="list">
          <thead><tr><th>Day</th><th class="num">Requests</th><th class="num">Prompt</th><th class="num">Generated</th></tr></thead>
          <tbody>
            {#each dayRows as r (r.day)}
              <tr>
                <td>{new Date(`${r.day}T12:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}</td>
                <td class="num">{r.requests.toLocaleString()}</td>
                <td class="num">{compactNumber(r.prompt)}</td>
                <td class="num">{compactNumber(r.generated)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  {:else if daily}
    <p class="sub">No traffic in the last {range} days yet.</p>
  {/if}
</section>

<style>
  .head {
    display: flex;
    align-items: center;
    gap: 0.6rem 0.9rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
  }
  .head h2 {
    margin: 0;
  }
  .spacer {
    flex: 1;
  }
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .tile {
    display: grid;
    gap: 0.1rem;
    padding: 0.7rem 0.9rem;
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .tile .label {
    font-size: 0.8rem;
    color: var(--muted);
  }
  .tile .value {
    font-size: 1.7rem;
    font-weight: 600;
  }
  .tile .sub {
    font-size: 0.75rem;
  }
  table.list {
    margin-top: 0.8rem;
  }
  .tables {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    margin-top: 0.8rem;
  }
  h3 {
    font-size: 0.95rem;
    margin: 0;
  }
  .scroll {
    max-height: 24rem;
    overflow: auto;
  }
</style>
