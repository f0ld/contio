<script lang="ts">
  import BarChart, { type Bar } from "$lib/components/BarChart.svelte";
  import ScatterChart, { type Point } from "$lib/components/ScatterChart.svelte";
  import {
    benchExportUrl,
    deleteBenchRun,
    errorText,
    fetchBenchRun,
    fetchBenchRuns,
    fetchModels,
    formatBytes,
    startBench,
  } from "$lib/daemon";
  import { jobs, refreshJobs } from "$lib/jobs.svelte";
  import { live } from "$lib/live.svelte";
  import type {
    BenchPreset,
    BenchRun,
    BenchSuite,
    ModelStatus,
    PerfReportView,
    QualityReportView,
    RetrievalReportView,
    SpeechReportView,
    ToolsReportView,
  } from "$lib/types";

  const SUITES: { id: BenchSuite; name: string; title: string }[] = [
    { id: "perf", name: "speed, energy, memory", title: "Prefill and generation speed, time to first token, power, load time, memory" },
    { id: "tools", name: "tool calling", title: "23 cases: simple, choosing, parallel, multi-turn, no-call, JSON; gates agent-ready" },
    { id: "reasoning", name: "reasoning", title: "Math word problems (GSM8K subset)" },
    { id: "knowledge", name: "knowledge", title: "Multiple choice across 14 subjects (MMLU-Pro subset)" },
    { id: "coding", name: "coding", title: "Python functions checked by unit tests in a podman sandbox (HumanEval subset)" },
    { id: "retrieval", name: "retrieval", title: "Ranking 48 documents for 24 paraphrased queries (embedding and reranker models)" },
    { id: "speech", name: "speech", title: "Word error rate on 20 LibriSpeech utterances (Whisper models)" },
  ];
  const QUALITY: BenchSuite[] = ["reasoning", "knowledge", "coding"];

  let models = $state<ModelStatus[]>([]);
  let runs = $state<BenchRun[]>([]);
  let error = $state<string | null>(null);

  // run form
  let picked = $state<string[]>([]);
  let preset = $state<BenchPreset>("quick");
  let suites = $state<BenchSuite[]>(["perf"]);
  let exclusive = $state(true);
  let thinkingOff = $state(false);
  let profile = $state<"" | "power-saver" | "balanced" | "performance">("");
  let jobId = $state<number | null>(null);
  let job = $derived(jobs.byId(jobId) ?? jobs.list.find((j) => j.kind === "bench" && j.state === "running") ?? null);

  let selected = $state<BenchRun | null>(null);

  async function load() {
    try {
      const [m, r] = await Promise.all([fetchModels(), fetchBenchRuns()]);
      models = m.filter((x) => x.kind === "chat" || x.kind === "embedding" || x.kind === "reranker" || x.kind === "transcription");
      runs = r;
      error = null;
    } catch (e) {
      error = errorText(e);
    }
  }

  $effect(() => {
    if (live.online) load();
  });

  // refresh results whenever a benchmark job changes state
  let lastState = "";
  $effect(() => {
    const s = job ? `${job.id}:${job.state}:${job.step}` : "";
    if (s !== lastState) {
      lastState = s;
      if (job && job.state !== "running") load();
    }
  });

  async function run() {
    error = null;
    try {
      jobId = (await startBench({ models: picked, preset, suites, exclusive, reload: true, power_profile: profile || undefined, thinking: thinkingOff ? false : undefined })).job_id;
      refreshJobs();
    } catch (e) {
      error = errorText(e);
    }
  }

  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  async function open(r: BenchRun) {
    try {
      selected = await fetchBenchRun(r.id);
    } catch (e) {
      error = errorText(e);
    }
  }

  async function remove(r: BenchRun) {
    try {
      await deleteBenchRun(r.id);
      if (selected?.id === r.id) selected = null;
      await load();
    } catch (e) {
      error = errorText(e);
    }
  }

  const label = (r: BenchRun) => r.model;
  const sub = (r: BenchRun) =>
    [
      r.target.device?.toUpperCase(),
      r.target.origin?.quant !== "none" ? r.target.origin?.quant : null,
      r.target.thinking === false ? "thinking off" : null,
      r.conditions.power_profile,
      r.preset,
    ]
      .filter(Boolean)
      .join(" · ");

  /** Latest finished run per model and suite. */
  function latest(suite: BenchSuite): BenchRun[] {
    const seen = new Set<string>();
    return runs.filter((r) => r.suite === suite && r.status === "done" && !seen.has(r.model) && seen.add(r.model));
  }

  function bars(list: BenchRun[], metric: string, scale = 1): Bar[] {
    return list
      .filter((r) => r.metrics[metric] != null)
      .map((r) => ({ label: label(r), sub: sub(r), value: r.metrics[metric] * scale }));
  }

  let perfRuns = $derived(latest("perf"));
  let toolRuns = $derived(latest("tools"));
  let qualityRuns = $derived(Object.fromEntries(QUALITY.map((s) => [s, latest(s)])) as Record<BenchSuite, BenchRun[]>);
  /** Models with a speed and a tool-calling result. */
  let tradeoff = $derived<Point[]>(
    perfRuns.flatMap((p) => {
      const t = toolRuns.find((r) => r.model === p.model);
      const x = p.metrics.tg_tps;
      const y = t?.metrics.tools_pass_rate;
      return t && x != null && y != null ? [{ label: p.model, sub: sub(p), x, y: y * 100 }] : [];
    }),
  );
  let retrievalRuns = $derived(latest("retrieval"));
  let speechRuns = $derived(latest("speech"));
  let anyQuality = $derived(toolRuns.length > 0 || retrievalRuns.length > 0 || speechRuns.length > 0 || QUALITY.some((s) => qualityRuns[s]?.length));
  let profiles = $derived(new Set(perfRuns.map((r) => r.conditions.power_profile)));

  const memOf = (r: BenchRun) =>
    r.target.device === "gpu" ? r.metrics.mem_gpu_bytes : Math.max(r.metrics.mem_system_bytes ?? 0, r.metrics.rss_bytes ?? 0) || undefined;
  const when = (s: number) => new Date(s * 1000).toLocaleString();
  const n = (v: number | null | undefined, d = 1) => (v == null ? "–" : v.toFixed(d));
  const pct = (v: number | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);
  let perfReport = $derived(selected?.suite === "perf" ? (selected.report as PerfReportView | null) : null);
  let toolsReport = $derived(selected?.suite === "tools" ? (selected.report as ToolsReportView | null) : null);
  let speechReport = $derived(selected?.suite === "speech" ? (selected.report as SpeechReportView | null) : null);
  let retrievalReport = $derived(selected?.suite === "retrieval" ? (selected.report as RetrievalReportView | null) : null);
  let qualityReport = $derived(
    selected && QUALITY.includes(selected.suite) ? (selected.report as QualityReportView | null) : null,
  );
</script>

<section class="card form">
  <h2>Run a benchmark</h2>
  <div class="row">
    <span class="label">Models</span>
    {#each models as m (m.id)}
      <label class="pick"><input type="checkbox" checked={picked.includes(m.id)} onchange={() => (picked = toggle(picked, m.id))} /> {m.id} <span class="sub">{m.device.toUpperCase()}</span></label>
    {/each}
    {#if models.length === 0}<span class="sub">no chat models installed</span>{/if}
  </div>
  <div class="row">
    <span class="label">Measure</span>
    {#each SUITES as su (su.id)}
      <label class="pick" title={su.title}><input type="checkbox" checked={suites.includes(su.id)} onchange={() => (suites = toggle(suites, su.id))} /> {su.name}</label>
    {/each}
  </div>
  <div class="row">
    <span class="label">Preset</span>
    <select bind:value={preset}>
      <option value="quick">quick: short prompts; 20 items per quality suite</option>
      <option value="standard">standard: + 4k prompts, 3 runs; all items</option>
      <option value="full">full: + 16k prompts, 5 runs; all items 3×</option>
    </select>
    <span class="label" title="Switched for the run and restored afterwards">Power profile</span>
    <select bind:value={profile}>
      <option value="">as it is</option>
      <option value="power-saver">power saver</option>
      <option value="balanced">balanced</option>
      <option value="performance">performance</option>
    </select>
    <label class="pick" title="Ask Qwen-style models to answer without thinking (tool calling and quality suites)"><input type="checkbox" bind:checked={thinkingOff} /> thinking off</label>
    <label class="pick" title="Unload other idle models so they don't share memory bandwidth"><input type="checkbox" bind:checked={exclusive} /> unload other models</label>
    <span class="spacer"></span>
    <button class="primary" disabled={!picked.length || !suites.length || job?.state === "running"} onclick={run}>
      {job?.state === "running" ? "Running…" : "Run"}
    </button>
  </div>
  {#if job?.state === "running"}
    <p class="sub">job #{job.id}: {job.step ?? "starting"} · <a href="/jobs">details</a></p>
  {:else if job?.error}
    <p class="error-text">{job.error}</p>
  {/if}
  {#if error}<p class="error-text">{error}</p>{/if}
</section>

{#if perfRuns.length}
  <section class="card">
    <div class="head">
      <h2>Speed and energy</h2>
      <span class="sub">latest run per model</span>
    </div>
    {#if profiles.size > 1}
      <p class="warn">Runs used different power profiles ({[...profiles].join(", ")}); compare them with care.</p>
    {/if}
    <div class="charts">
      <BarChart title="Generation" unit="tok/s" bars={bars(perfRuns, "tg_tps")} />
      <BarChart title="Prompt processing, 512 tokens" unit="tok/s" digits={0} bars={bars(perfRuns, "pp512_tps")} />
      <BarChart title="Prompt processing, 4k tokens" unit="tok/s" digits={0} bars={bars(perfRuns, "pp4096_tps")} />
      <BarChart title="Time to first token, 512 tokens" unit="s" digits={2} lowerIsBetter bars={bars(perfRuns, "ttft512_s")} />
      <BarChart title="Energy per generated token" unit="J" digits={2} lowerIsBetter bars={bars(perfRuns, "j_per_token")} />
      <BarChart title="Load time" unit="s" lowerIsBetter bars={bars(perfRuns, "load_s")} />
    </div>
  </section>
{/if}

{#if anyQuality}
  <section class="card">
    <div class="head">
      <h2>Answer quality</h2>
      <span class="sub">share of items passed · agent-ready = tool calling ≥ 90% overall and ≥ 75% in every category</span>
    </div>
    <div class="charts">
      {#if tradeoff.length}
        <ScatterChart
          title="Speed vs tool calling"
          points={tradeoff}
          xLabel="generation tok/s"
          yLabel="tool calls passed"
          yMax={100}
          yFormat={(v) => `${Math.round(v)}%`}
        />
      {/if}
      {#if toolRuns.length}
        <BarChart title="Tool calling" bars={bars(toolRuns, "tools_pass_rate", 100)} format={(v) => `${Math.round(v)}%`} />
      {/if}
      {#if speechRuns.length}
        <BarChart title="Speech: word error rate" lowerIsBetter bars={bars(speechRuns, "speech_wer", 100)} format={(v) => `${v.toFixed(1)}%`} />
      {/if}
      {#if retrievalRuns.length}
        <BarChart title="Retrieval: relevant document ranked first" bars={bars(retrievalRuns, "retrieval_recall1", 100)} format={(v) => `${Math.round(v)}%`} />
      {/if}
      {#each QUALITY as q (q)}
        {#if qualityRuns[q]?.length}
          <BarChart
            title={SUITES.find((s) => s.id === q)?.title ?? q}
            bars={bars(qualityRuns[q], `${q}_pass_rate`, 100)}
            format={(v) => `${Math.round(v)}%`}
          />
        {/if}
      {/each}
    </div>
  </section>
{/if}

<section class="card">
  <div class="head">
    <h2>All runs</h2>
    <span class="spacer"></span>
    <a class="small" href={benchExportUrl("csv")} download="contio-bench.csv">Export CSV</a>
    <a class="small" href={benchExportUrl("json")} download="contio-bench.json">JSON</a>
  </div>
  {#if runs.length === 0}
    <p class="sub">No results yet. Pick models above and press Run.</p>
  {:else}
    <table class="list">
      <thead>
        <tr><th>#</th><th>Model</th><th>Setup</th><th>Suite</th><th class="num">Result</th><th>When</th><th></th></tr>
      </thead>
      <tbody>
        {#each runs as r (r.id)}
          <tr class="clickable" class:selected={selected?.id === r.id} onclick={() => open(r)}>
            <td class="sub">{r.id}</td>
            <td><b>{r.model}</b></td>
            <td class="sub">{sub(r)}</td>
            <td>{r.suite === "perf" ? "speed" : r.suite === "tools" ? "tool calling" : r.suite}</td>
            <td class="num">
              {#if r.status !== "done"}<span class="state {r.status}">{r.status}</span>
              {:else if r.suite === "perf"}{n(r.metrics.tg_tps)} tok/s · {n(r.metrics.pp512_tps, 0)} pp
              {:else if r.suite === "speech"}WER {n(r.metrics.speech_wer * 100)}% · RTF {n(r.metrics.speech_rtf, 3)}
              {:else if r.suite === "retrieval"}{pct(r.metrics.retrieval_recall1)} first · MRR {n(r.metrics.retrieval_mrr, 2)}
              {:else if r.suite === "tools"}{pct(r.metrics.tools_pass_rate)}{#if r.metrics.agent_ready === 1}<span class="pill fits">agent-ready</span>{/if}
              {:else}{pct(r.metrics[`${r.suite}_pass_rate`])}{/if}
            </td>
            <td class="sub">{when(r.started_at)}</td>
            <td><button class="small danger" title="Delete" disabled={r.status === "running"} onclick={(e) => { e.stopPropagation(); remove(r); }}>✕</button></td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>

{#if selected}
  <section class="card detail">
    <div class="head">
      <h2>Run {selected.id}: {selected.model}</h2>
      <span class="sub">{sub(selected)} · {selected.target.backend}</span>
      <span class="spacer"></span>
      <button class="small" onclick={() => (selected = null)}>Close</button>
    </div>
    <p class="sub">
      {selected.conditions.on_ac === false ? `on battery (${selected.conditions.battery_percent}%)` : "on AC"}
      · {selected.conditions.backend}
      {#if selected.conditions.other_models?.length}· also loaded: {selected.conditions.other_models.join(", ")}{/if}
    </p>
    {#if selected.error}<p class="error-text">{selected.error}</p>{/if}

    {#if perfReport}
      <p>
        {#if perfReport.load.load_s != null}Load {n(perfReport.load.load_s)} s ·{/if}
        {#if memOf(selected)}memory {formatBytes(memOf(selected) ?? 0)}{/if}
        {#if perfReport.load.estimated_bytes}(estimated {formatBytes(perfReport.load.estimated_bytes)}){/if}
        {#if perfReport.perf.idle_watts != null}· idle {n(perfReport.perf.idle_watts)} W{/if}
      </p>
      <table class="list">
        <thead>
          <tr><th class="num">Prompt</th><th class="num">TTFT s</th><th class="num">Prefill tok/s</th><th class="num">Decode tok/s</th><th class="num">Watts</th><th class="num">J/token</th></tr>
        </thead>
        <tbody>
          {#each perfReport.perf.prompts as p (p.target_tokens)}
            <tr>
              <td class="num">{p.summary.prompt_tokens}</td>
              <td class="num">{n(p.summary.ttft_s, 2)}</td>
              <td class="num">{n(p.summary.prefill_tps, 0)}</td>
              <td class="num">{n(p.summary.decode_tps)}</td>
              <td class="num">{n(p.summary.avg_watts)}</td>
              <td class="num">{n(p.summary.joules_per_token, 2)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      {#if perfReport.perf.concurrency}
        {@const c = perfReport.perf.concurrency}
        <p class="sub">{c.requests} parallel requests: {n(c.aggregate_tps)} tok/s total, {n(c.per_request_decode_tps)} tok/s per request while generating</p>
      {/if}
      {#if perfReport.perf.skipped.length}<p class="sub">Skipped (longer than the context): {perfReport.perf.skipped.join(", ")} tokens</p>{/if}
    {/if}

    {#if toolsReport}
      <p>
        <b>{pct(toolsReport.pass_rate)} passed</b>
        {#if toolsReport.agent_ready}<span class="pill fits">agent-ready</span>{:else}<span class="pill tight">not agent-ready</span>{/if}
        <span class="sub">{Object.entries(toolsReport.categories).map(([k, v]) => `${k.replace("_", "-")} ${pct(v)}`).join(" · ")}</span>
      </p>
      <table class="list">
        <thead><tr><th>Case</th><th>Category</th><th class="num">Passed</th><th class="num">Sec</th><th>First failure</th></tr></thead>
        <tbody>
          {#each toolsReport.cases as c (c.id)}
            <tr>
              <td>{c.id}</td>
              <td class="sub">{c.category}</td>
              <td class="num">{c.passed}/{c.runs}</td>
              <td class="num">{n(c.mean_s)}</td>
              <td class="sub why">{c.failures[0] ?? ""}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if speechReport}
      <p>
        <b>WER {n(speechReport.wer * 100)}%</b>
        <span class="sub">real-time factor {n(speechReport.rtf, 3)} (below 1 = faster than real time)</span>
      </p>
      <table class="list">
        <thead><tr><th>Utterance</th><th class="num">Errors</th><th>Transcript</th></tr></thead>
        <tbody>
          {#each speechReport.utterances.filter((u) => u.errors > 0) as u (u.id)}
            <tr>
              <td class="sub">{u.id}</td>
              <td class="num">{u.errors}/{u.words}</td>
              <td><div>{u.hypothesis}</div><div class="sub">{u.reference.toLowerCase()}</div></td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}

    {#if retrievalReport}
      <p>
        <b>{pct(retrievalReport.recall_at_1)} ranked first</b>
        <span class="sub">MRR {n(retrievalReport.mrr, 3)} · {retrievalReport.mode} · {n(retrievalReport.seconds)} s</span>
      </p>
      {#if retrievalReport.queries.some((q) => q.rank > 1)}
        <table class="list">
          <thead><tr><th>Missed query</th><th class="num">Rank</th><th>Ranked first instead</th></tr></thead>
          <tbody>
            {#each retrievalReport.queries.filter((q) => q.rank > 1) as q (q.query)}
              <tr><td>{q.query}</td><td class="num">{q.rank}</td><td class="sub">{q.top}</td></tr>
            {/each}
          </tbody>
        </table>
      {/if}
    {/if}

    {#if qualityReport}
      <p>
        <b>{pct(qualityReport.pass_rate)} passed</b>
        <span class="sub">
          {qualityReport.items.length} items × {qualityReport.reps}
          {#if qualityReport.truncated}· {qualityReport.truncated} hit the token limit{/if}
        </span>
      </p>
      {#if Object.keys(qualityReport.categories).length}
        <p class="sub">{Object.entries(qualityReport.categories).map(([k, v]) => `${k} ${pct(v)}`).join(" · ")}</p>
      {/if}
      <table class="list">
        <thead><tr><th>Item</th><th class="num">Passed</th><th class="num">Sec</th><th>First failure</th></tr></thead>
        <tbody>
          {#each qualityReport.items as it (it.id)}
            <tr>
              <td>{it.id}{#if it.category}<span class="sub"> · {it.category}</span>{/if}</td>
              <td class="num">{it.passed}/{it.runs}</td>
              <td class="num">{n(it.mean_s)}</td>
              <td class="sub why">{it.failures[0] ?? ""}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
{/if}

<style>
  .form {
    display: grid;
    gap: 0.6rem;
  }
  .form h2 {
    margin: 0;
  }
  .row,
  .head {
    display: flex;
    align-items: center;
    gap: 0.5rem 0.9rem;
    flex-wrap: wrap;
  }
  .head h2 {
    margin: 0;
  }
  .head {
    margin-bottom: 0.6rem;
  }
  .label {
    color: var(--muted);
    font-size: 0.85rem;
  }
  .pick {
    font-size: 0.9rem;
    display: flex;
    gap: 0.3rem;
    align-items: center;
  }
  .spacer {
    flex: 1;
  }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 1.25rem 2rem;
  }
  .warn {
    color: var(--warn);
    font-size: 0.9rem;
  }
  tr.clickable {
    cursor: pointer;
  }
  tr.clickable:hover td,
  tr.selected td {
    background: var(--bg);
  }
  .state.failed {
    color: var(--err);
  }
  .state.running {
    color: var(--accent);
  }
  .why {
    max-width: 28rem;
    overflow-wrap: anywhere;
  }
  .pill {
    margin-left: 0.4rem;
  }
  a.small {
    font-size: 0.85rem;
  }
</style>
