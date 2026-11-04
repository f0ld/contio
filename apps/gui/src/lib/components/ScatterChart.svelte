<script lang="ts" module>
  export interface Point {
    label: string;
    sub?: string;
    x: number;
    y: number;
  }
</script>

<script lang="ts">
  /** Two measures per model: one dot each, direct-labeled, tooltip on hover. */
  let {
    title,
    points,
    xLabel,
    yLabel,
    xFormat = (v: number) => `${v}`,
    yFormat = (v: number) => `${v}`,
    yMax,
  }: {
    title: string;
    points: Point[];
    xLabel: string;
    yLabel: string;
    xFormat?: (v: number) => string;
    yFormat?: (v: number) => string;
    yMax?: number;
  } = $props();

  const W = 460;
  const H = 260;
  const PAD = { left: 44, right: 16, top: 12, bottom: 36 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  function nice(max: number): number[] {
    const step0 = (max || 1) / 4;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= step0) ?? step0;
    const out = [];
    for (let t = 0; t <= max + step * 0.999; t += step) out.push(Number(t.toPrecision(6)));
    return out;
  }

  let xTicks = $derived(nice(Math.max(...points.map((p) => p.x), 0)));
  let yTicks = $derived(nice(yMax ?? Math.max(...points.map((p) => p.y), 0)));
  let xMaxT = $derived(xTicks[xTicks.length - 1] || 1);
  let yMaxT = $derived(yTicks[yTicks.length - 1] || 1);
  const sx = (v: number) => PAD.left + (v / xMaxT) * plotW;
  const sy = (v: number) => PAD.top + plotH - (v / yMaxT) * plotH;
  let hover = $state<number | null>(null);

  /** Which points get a direct label: skip any that would overlap one already placed. */
  let labeled = $derived.by(() => {
    const boxes: { x0: number; x1: number; y0: number; y1: number }[] = [];
    const keep = new Set<number>();
    const order = points.map((_, i) => i).sort((a, b) => points[b].y - points[a].y);
    for (const i of order) {
      const p = points[i];
      const w = p.label.length * 6.2 + 9;
      const right = sx(p.x) > W - 140;
      const x0 = right ? sx(p.x) - w : sx(p.x);
      const box = { x0, x1: x0 + w, y0: sy(p.y) - 8, y1: sy(p.y) + 8 };
      const dotsHit = points.some((q, j) => j !== i && sx(q.x) > box.x0 && sx(q.x) < box.x1 && Math.abs(sy(q.y) - sy(p.y)) < 8);
      if (!dotsHit && !boxes.some((b) => b.x0 < box.x1 && box.x0 < b.x1 && b.y0 < box.y1 && box.y0 < b.y1)) {
        boxes.push(box);
        keep.add(i);
      }
    }
    return keep;
  });
</script>

<figure class="chart">
  <figcaption><b>{title}</b></figcaption>
  {#if points.length === 0}
    <p class="sub">Needs a speed run and a tool-calling run of the same model.</p>
  {:else}
    <svg viewBox="0 0 {W} {H}" role="img" aria-label={title}>
      {#each yTicks as t (t)}
        <line class="grid" x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} />
        <text class="tick" x={PAD.left - 6} y={sy(t) + 3} text-anchor="end">{yFormat(t)}</text>
      {/each}
      {#each xTicks as t (t)}
        <text class="tick" x={sx(t)} y={H - PAD.bottom + 14} text-anchor="middle">{xFormat(t)}</text>
      {/each}
      <text class="axis" x={PAD.left + plotW / 2} y={H - 4} text-anchor="middle">{xLabel}</text>
      <text class="axis" x={12} y={PAD.top + plotH / 2} text-anchor="middle" transform="rotate(-90 12 {PAD.top + plotH / 2})">{yLabel}</text>
      {#each points as p, i (p.label + (p.sub ?? ""))}
        <g
          role="img"
          aria-label="{p.label}: {xFormat(p.x)} {xLabel}, {yFormat(p.y)} {yLabel}"
          class:dim={hover !== null && hover !== i}
          onpointerenter={() => (hover = i)}
          onpointerleave={() => (hover = null)}
        >
          <circle class="hit" cx={sx(p.x)} cy={sy(p.y)} r="12" />
          <circle class="dot" cx={sx(p.x)} cy={sy(p.y)} r="5" />
          {#if labeled.has(i) || hover === i}
            <text class="label" x={sx(p.x) + (sx(p.x) > W - 140 ? -9 : 9)} y={sy(p.y) + 4} text-anchor={sx(p.x) > W - 140 ? "end" : "start"}>
              {p.label}
            </text>
          {/if}
        </g>
      {/each}
    </svg>
    {#if hover !== null && points[hover]}
      {@const p = points[hover]}
      <div class="tooltip">
        <b>{xFormat(p.x)} · {yFormat(p.y)}</b>
        <span>{p.label}{p.sub ? ` · ${p.sub}` : ""}</span>
      </div>
    {/if}
  {/if}
</figure>

<style>
  .chart {
    position: relative;
    margin: 0;
  }
  figcaption {
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
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
  .tick,
  .axis {
    font-size: 10px;
    fill: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .axis {
    font-size: 11px;
  }
  .dot {
    fill: var(--accent);
    stroke: var(--card);
    stroke-width: 2;
  }
  .hit {
    fill: transparent;
  }
  .label {
    font-size: 11px;
    fill: var(--text);
  }
  .dim .dot {
    opacity: 0.45;
  }
  .tooltip {
    position: absolute;
    top: 1.6rem;
    right: 0;
    pointer-events: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.55rem;
    font-size: 0.8rem;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.12);
    display: grid;
    white-space: nowrap;
  }
  .tooltip span {
    color: var(--muted);
  }
</style>
