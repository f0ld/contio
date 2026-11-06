<script lang="ts">
  /** One measure over time: 2px line with a light area wash, crosshair tooltip, current value at the end. */
  let {
    points,
    unit = "",
    height = 180,
    format = (v: number) => v.toFixed(0),
    timeLabel = (t: number) => new Date(t * 1000).toLocaleTimeString(),
  }: {
    points: { t: number; v: number }[];
    unit?: string;
    height?: number;
    format?: (v: number) => string;
    timeLabel?: (t: number) => string;
  } = $props();

  const W = 900;
  const PAD = { left: 44, right: 64, top: 10, bottom: 22 };
  let plotW = W - PAD.left - PAD.right;
  let plotH = $derived(height - PAD.top - PAD.bottom);

  function nice(max: number): number[] {
    if (max <= 0) return [0, 1];
    const step0 = max / 4;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= step0) ?? step0;
    const out = [];
    for (let t = 0; t <= max + step * 0.999; t += step) out.push(Number(t.toPrecision(6)));
    return out;
  }

  let ticks = $derived(nice(Math.max(...points.map((p) => p.v), 0)));
  let yMax = $derived(ticks[ticks.length - 1] || 1);
  let t0 = $derived(points[0]?.t ?? 0);
  let t1 = $derived(points[points.length - 1]?.t ?? 1);
  const sx = (t: number) => PAD.left + ((t - t0) / Math.max(1, t1 - t0)) * plotW;
  const sy = (v: number) => PAD.top + plotH - (v / yMax) * plotH;

  let line = $derived(points.map((p, i) => `${i ? "L" : "M"}${sx(p.t).toFixed(1)},${sy(p.v).toFixed(1)}`).join(" "));
  let area = $derived(points.length ? `${line} L${sx(t1)},${sy(0)} L${sx(t0)},${sy(0)} Z` : "");
  let last = $derived(points[points.length - 1]);

  let hover = $state<number | null>(null);
  let svg: SVGSVGElement | undefined = $state();

  function move(e: PointerEvent) {
    if (!svg || !points.length) return;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const t = t0 + ((x - PAD.left) / plotW) * (t1 - t0);
    let best = 0;
    for (let i = 1; i < points.length; i++) if (Math.abs(points[i].t - t) < Math.abs(points[best].t - t)) best = i;
    hover = best;
  }

  // a few time ticks: start, quarters, now
  let timeTicks = $derived([0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(t0 + f * (t1 - t0))));
</script>

<div class="wrap">
  <svg
    bind:this={svg}
    viewBox="0 0 {W} {height}"
    role="img"
    aria-label="throughput over time"
    onpointermove={move}
    onpointerleave={() => (hover = null)}
  >
    {#each ticks as y (y)}
      <line class="grid" x1={PAD.left} x2={W - PAD.right} y1={sy(y)} y2={sy(y)} />
      <text class="tick" x={PAD.left - 6} y={sy(y) + 3} text-anchor="end">{format(y)}</text>
    {/each}
    {#each timeTicks as t, i (i)}
      <text class="tick" x={sx(t)} y={height - 6} text-anchor={i === 0 ? "start" : i === 4 ? "end" : "middle"}>
        {i === 4 ? "now" : `−${Math.round((t1 - t) / 60 * 10) / 10} min`}
      </text>
    {/each}
    <path class="area" d={area} />
    <path class="line" d={line} />
    {#if last}
      <circle class="dot" cx={sx(last.t)} cy={sy(last.v)} r="4" />
      <text class="end" x={sx(last.t) + 8} y={sy(last.v) + 4}>{format(last.v)}{unit ? ` ${unit}` : ""}</text>
    {/if}
    {#if hover !== null && points[hover]}
      {@const p = points[hover]}
      <line class="cross" x1={sx(p.t)} x2={sx(p.t)} y1={PAD.top} y2={PAD.top + plotH} />
      <circle class="dot" cx={sx(p.t)} cy={sy(p.v)} r="4" />
    {/if}
  </svg>
  {#if hover !== null && points[hover]}
    {@const p = points[hover]}
    <div class="tooltip" style:left="{(sx(p.t) / W) * 100}%">
      <b>{format(p.v)}{unit ? ` ${unit}` : ""}</b>
      <span>{timeLabel(p.t)}</span>
    </div>
  {/if}
</div>

<style>
  .wrap {
    position: relative;
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
  .area {
    fill: var(--series-1);
    opacity: 0.12;
  }
  .line {
    fill: none;
    stroke: var(--series-1);
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .dot {
    fill: var(--series-1);
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
    top: 0;
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
