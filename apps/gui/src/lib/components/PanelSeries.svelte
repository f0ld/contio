<script lang="ts" module>
  export interface Line {
    label: string;
    /** CSS color of the series (identity) */
    color: string;
    /** one value per entry of `times`; null = no reading */
    values: (number | null)[];
  }
  export interface Panel {
    title: string;
    unit: string;
    lines: Line[];
    /** fixed top of the scale (e.g. pool size); else from the data */
    max?: number;
    format: (v: number) => string;
  }
</script>

<script lang="ts">
  /**
   * Several measures over one shared time axis: one panel per unit (no dual axis),
   * 2px lines, direct labels at the end, one crosshair and tooltip across all panels.
   */
  let {
    times,
    panels,
    legend,
    window,
  }: {
    times: number[];
    panels: Panel[];
    legend: { label: string; color: string }[];
    /** fixed time range [from, to] in unix seconds, to line up with other charts */
    window?: [number, number];
  } = $props();

  const W = 900;
  const PANEL_H = 110;
  const GAP = 36;
  // same plot edges as TimeSeries, so stacked charts line up
  const PAD = { left: 44, right: 64, top: 22, bottom: 22 };
  const plotW = W - PAD.left - PAD.right;
  let H = $derived(PAD.top + panels.length * PANEL_H + (panels.length - 1) * GAP + PAD.bottom);

  function nice(max: number): number[] {
    if (max <= 0) return [0, 1];
    const step0 = max / 3;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= step0) ?? step0;
    const out = [];
    for (let t = 0; t <= max + step * 0.999; t += step) out.push(Number(t.toPrecision(6)));
    return out;
  }

  let t0 = $derived(window?.[0] ?? times[0] ?? 0);
  let t1 = $derived(window?.[1] ?? times[times.length - 1] ?? 1);
  const sx = (t: number) => PAD.left + ((t - t0) / Math.max(1, t1 - t0)) * plotW;

  let laidOut = $derived(
    panels.map((p, i) => {
      const top = PAD.top + i * (PANEL_H + GAP);
      const peak = Math.max(0, ...p.lines.flatMap((l) => l.values.filter((v): v is number => v != null)));
      const ticks = nice(p.max ?? peak);
      const yMax = ticks[ticks.length - 1] || 1;
      const sy = (v: number) => top + PANEL_H - (Math.min(v, yMax) / yMax) * PANEL_H;
      const lines = p.lines.map((l) => {
        // break the line where readings are missing
        let d = "";
        let pen = false;
        l.values.forEach((v, j) => {
          if (v == null) return void (pen = false);
          d += `${pen ? "L" : "M"}${sx(times[j]).toFixed(1)},${sy(v).toFixed(1)} `;
          pen = true;
        });
        const lastIdx = l.values.findLastIndex((v) => v != null);
        return { ...l, d, lastIdx, lastY: lastIdx >= 0 ? sy(l.values[lastIdx]!) : 0 };
      });
      // end labels: keep at least 14px apart
      const labels = lines.filter((l) => l.lastIdx >= 0).sort((a, b) => a.lastY - b.lastY).map((l) => ({ l, y: l.lastY + 4 }));
      for (let k = 1; k < labels.length; k++) labels[k].y = Math.max(labels[k].y, labels[k - 1].y + 14);
      return { ...p, top, ticks, sy, lines, labels };
    }),
  );

  let hover = $state<number | null>(null);
  let svg: SVGSVGElement | undefined = $state();

  function move(e: PointerEvent) {
    if (!svg || !times.length) return;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const t = t0 + ((x - PAD.left) / plotW) * (t1 - t0);
    let best = 0;
    for (let i = 1; i < times.length; i++) if (Math.abs(times[i] - t) < Math.abs(times[best] - t)) best = i;
    hover = best;
  }

  let timeTicks = $derived([0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(t0 + f * (t1 - t0))));
  const show = (p: Panel, v: number | null) => (v == null ? "–" : `${p.format(v)} ${p.unit}`);
</script>

<div class="wrap">
  <div class="legend">
    {#each legend as l (l.label)}<span><i style:background={l.color}></i>{l.label}</span>{/each}
  </div>
  <svg
    bind:this={svg}
    viewBox="0 0 {W} {H}"
    role="img"
    aria-label={panels.map((p) => p.title).join(" and ") + " over time"}
    onpointermove={move}
    onpointerleave={() => (hover = null)}
  >
    {#each laidOut as p (p.title)}
      <text class="title" x={PAD.left} y={p.top - 12}>{p.title} ({p.unit})</text>
      {#each p.ticks as y (y)}
        <line class="grid" x1={PAD.left} x2={W - PAD.right} y1={p.sy(y)} y2={p.sy(y)} />
        <text class="tick" x={PAD.left - 6} y={p.sy(y) + 3} text-anchor="end">{p.format(y)}</text>
      {/each}
      {#each p.lines as l (l.label)}
        <path class="line" d={l.d} style:stroke={l.color} />
        {#if l.lastIdx >= 0}
          <circle class="dot" cx={sx(times[l.lastIdx])} cy={l.lastY} r="4" style:fill={l.color} />
        {/if}
      {/each}
      {#each p.labels as { l, y } (l.label)}
        <text class="end" x={W - PAD.right + 10} {y}>{l.label} {p.format(l.values[l.lastIdx]!)}</text>
      {/each}
      {#if hover !== null}
        {#each p.lines as l (l.label)}
          {#if l.values[hover] != null}
            <circle class="dot" cx={sx(times[hover])} cy={p.sy(l.values[hover]!)} r="4" style:fill={l.color} />
          {/if}
        {/each}
      {/if}
    {/each}
    {#each timeTicks as t, i (i)}
      <text class="tick" x={sx(t)} y={H - 6} text-anchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}>
        {i === 4 ? "now" : `−${Math.round(((t1 - t) / 60) * 10) / 10} min`}
      </text>
    {/each}
    {#if hover !== null}
      <line class="cross" x1={sx(times[hover])} x2={sx(times[hover])} y1={PAD.top} y2={H - PAD.bottom} />
    {/if}
  </svg>
  {#if hover !== null}
    {@const h = hover}
    <div class="tooltip" style:left="{(sx(times[h]) / W) * 100}%">
      <span class="time">{new Date(times[h] * 1000).toLocaleTimeString()}</span>
      {#each panels as p (p.title)}
        {#each p.lines as l (l.label)}
          <span class="row"><i style:background={l.color}></i>{l.label}<b>{show(p, l.values[h])}</b></span>
        {/each}
      {/each}
    </div>
  {/if}
</div>

<style>
  .wrap {
    position: relative;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 1rem;
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
  }
  .legend span,
  .row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .legend i,
  .row i {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    display: inline-block;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
    overflow: visible;
    touch-action: none;
  }
  .grid {
    stroke: var(--border);
    stroke-width: 1;
  }
  .tick {
    font-size: 11px;
    fill: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .title {
    font-size: 12px;
    fill: var(--muted);
  }
  .line {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .dot {
    stroke: var(--card);
    stroke-width: 2;
  }
  .end {
    font-size: 12px;
    fill: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .cross {
    stroke: var(--muted);
    stroke-width: 1;
  }
  .tooltip {
    position: absolute;
    top: 1.5rem;
    transform: translateX(-50%);
    pointer-events: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.55rem;
    font-size: 0.8rem;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.15);
    display: grid;
    gap: 0.1rem;
    white-space: nowrap;
  }
  .time {
    color: var(--muted);
  }
  .row b {
    margin-left: auto;
    padding-left: 0.8rem;
    font-variant-numeric: tabular-nums;
  }
</style>
