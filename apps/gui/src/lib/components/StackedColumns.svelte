<script lang="ts" module>
  export interface Column {
    /** x label, e.g. a date */
    key: string;
    label: string;
    /** series name → value */
    values: Record<string, number>;
  }
</script>

<script lang="ts">
  /** Part-to-whole per column (e.g. tokens per day by model): stacked, 2px gaps, legend, per-segment tooltip. */
  let {
    columns,
    series,
    format = (v: number) => String(v),
  }: { columns: Column[]; series: string[]; format?: (v: number) => string } = $props();

  const W = 900;
  const H = 240;
  const PAD = { left: 48, right: 8, top: 10, bottom: 26 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const GAP = 2;

  const color = (i: number) => `var(--series-${i < 7 ? i + 1 : 8})`;

  function nice(max: number): number[] {
    if (max <= 0) return [0, 1];
    const step0 = max / 4;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= step0) ?? step0;
    const out = [];
    for (let t = 0; t <= max + step * 0.999; t += step) out.push(Number(t.toPrecision(6)));
    return out;
  }

  let totals = $derived(columns.map((c) => series.reduce((a, s) => a + (c.values[s] ?? 0), 0)));
  let ticks = $derived(nice(Math.max(...totals, 0)));
  let yMax = $derived(ticks[ticks.length - 1] || 1);
  let slot = $derived(plotW / Math.max(1, columns.length));
  let barW = $derived(Math.min(24, slot * 0.7));
  const sy = (v: number) => PAD.top + plotH - (v / yMax) * plotH;

  /** Segments bottom-up; the top one gets the rounded end. */
  let stacks = $derived(
    columns.map((c, ci) => {
      let base = 0;
      const segs = series
        .map((s, si) => ({ s, si, v: c.values[s] ?? 0 }))
        .filter((x) => x.v > 0)
        .map((x) => {
          const y0 = base;
          base += x.v;
          return { ...x, y0, y1: base };
        });
      return { c, ci, segs };
    }),
  );

  function segPath(x: number, w: number, yTop: number, yBottom: number, round: boolean): string {
    const h = yBottom - yTop;
    const r = round ? Math.min(4, w / 2, h) : 0;
    return `M${x},${yBottom} V${yTop + r} Q${x},${yTop} ${x + r},${yTop} H${x + w - r} Q${x + w},${yTop} ${x + w},${yTop + r} V${yBottom} Z`;
  }

  let hover = $state<{ ci: number; si: number } | null>(null);
  // label every column on short ranges, else about 8 labels
  let every = $derived(Math.max(1, Math.ceil(columns.length / 8)));
  const compact = (t: number) => (t >= 1e6 ? `${Number((t / 1e6).toPrecision(3))}M` : t >= 1000 ? `${Number((t / 1000).toPrecision(3))}K` : `${t}`);
</script>

<div class="wrap">
  {#if series.length > 1}
    <div class="legend">
      {#each series as s, i (s)}<span><i style:background={color(i)}></i>{s}</span>{/each}
    </div>
  {/if}
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="stacked columns">
    {#each ticks as t (t)}
      <line class="grid" x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} />
      <text class="tick" x={PAD.left - 6} y={sy(t) + 3} text-anchor="end">{compact(t)}</text>
    {/each}
    {#each stacks as { c, ci, segs } (c.key)}
      {@const x = PAD.left + ci * slot + (slot - barW) / 2}
      {#each segs as seg, k (seg.s)}
        {@const top = sy(seg.y1)}
        {@const bottom = sy(seg.y0) - (k > 0 ? GAP : 0)}
        {#if bottom - top > 0.5}
          <path
            class="seg"
            class:dim={hover !== null && !(hover.ci === ci && hover.si === seg.si)}
            d={segPath(x, barW, top, bottom, k === segs.length - 1)}
            style:fill={color(seg.si)}
            role="img"
            aria-label="{c.label}, {seg.s}: {format(seg.v)}"
            onpointerenter={() => (hover = { ci, si: seg.si })}
            onpointerleave={() => (hover = null)}
          />
        {/if}
      {/each}
      <!-- labels counted back from the newest column, evenly spaced -->
      {#if (columns.length - 1 - ci) % every === 0}
        <text class="tick" x={x + barW / 2} y={H - 8} text-anchor="middle">{c.label}</text>
      {/if}
    {/each}
  </svg>
  {#if hover}
    {@const c = columns[hover.ci]}
    {@const s = series[hover.si]}
    <div class="tooltip" style:left="{((PAD.left + hover.ci * slot + slot / 2) / W) * 100}%">
      <b>{format(c.values[s] ?? 0)}</b>
      <span>{s} · {c.label}</span>
      <span>day total {format(totals[hover.ci])}</span>
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
    color: var(--text);
    margin-bottom: 0.4rem;
  }
  .legend span {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .legend i {
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
  .seg {
    transition: opacity 0.1s;
  }
  .seg.dim {
    opacity: 0.5;
  }
  .tooltip {
    position: absolute;
    top: 1.6rem;
    transform: translateX(-50%);
    pointer-events: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.55rem;
    font-size: 0.8rem;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.15);
    display: grid;
    white-space: nowrap;
  }
  .tooltip span {
    color: var(--muted);
  }
</style>
