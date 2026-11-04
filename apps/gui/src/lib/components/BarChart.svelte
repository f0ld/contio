<script lang="ts" module>
  export interface Bar {
    label: string;
    sub?: string;
    value: number;
  }
</script>

<script lang="ts">
  /** One measure across models: horizontal bars, value at the tip, tooltip on hover/focus. */

  let {
    title,
    unit = "",
    bars,
    digits = 1,
    lowerIsBetter = false,
    format,
  }: {
    title: string;
    unit?: string;
    bars: Bar[];
    digits?: number;
    lowerIsBetter?: boolean;
    format?: (v: number) => string;
  } = $props();

  const ROW = 30;
  const BAR = 18;
  const LABEL_W = 170;
  const VALUE_W = 70;
  const WIDTH = 460;
  const plotW = WIDTH - LABEL_W - VALUE_W;

  let sorted = $derived([...bars].sort((a, b) => (lowerIsBetter ? a.value - b.value : b.value - a.value)));
  let max = $derived(Math.max(...bars.map((b) => b.value), 0) || 1);
  let ticks = $derived(niceTicks(max));
  let scaleMax = $derived(ticks[ticks.length - 1]);
  let height = $derived(sorted.length * ROW + 22);
  let hover = $state<number | null>(null);

  const text = (v: number) => (format ? format(v) : `${v.toFixed(digits)}${unit ? ` ${unit}` : ""}`);

  function niceTicks(m: number): number[] {
    const step0 = m / 4;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= step0) ?? step0;
    const out = [];
    for (let t = 0; t <= m + step * 0.999; t += step) out.push(Number(t.toPrecision(6)));
    return out;
  }

  /** Bar path: square at the baseline, 4px rounded data end. */
  function barPath(w: number, y: number): string {
    const r = Math.min(4, w / 2, BAR / 2);
    const x0 = LABEL_W;
    const x1 = LABEL_W + w;
    return `M${x0},${y} H${x1 - r} Q${x1},${y} ${x1},${y + r} V${y + BAR - r} Q${x1},${y + BAR} ${x1 - r},${y + BAR} H${x0} Z`;
  }

  const compact = (t: number) => (t >= 1000 ? `${Number((t / 1000).toPrecision(3))}K` : `${t}`);
</script>

<figure class="chart">
  <figcaption>
    <b>{title}</b>
    <span class="sub">{lowerIsBetter ? "lower is better" : "higher is better"}</span>
  </figcaption>
  {#if sorted.length === 0}
    <p class="sub empty">no data yet</p>
  {:else}
    <svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label={title}>
      {#each ticks as t (t)}
        {@const x = LABEL_W + (t / scaleMax) * plotW}
        <line class="grid" x1={x} x2={x} y1="0" y2={sorted.length * ROW} />
        <text class="tick" x={x} y={sorted.length * ROW + 15} text-anchor="middle">{format ? format(t) : compact(t)}</text>
      {/each}
      {#each sorted as b, i (b.label + (b.sub ?? ""))}
        {@const y = i * ROW + (ROW - BAR) / 2}
        {@const w = Math.max(2, (b.value / scaleMax) * plotW)}
        <g
          class="row"
          class:dim={hover !== null && hover !== i}
          role="img"
          aria-label="{b.label}: {text(b.value)}"
          onpointerenter={() => (hover = i)}
          onpointerleave={() => (hover = null)}
        >
          <rect class="hit" x="0" y={i * ROW} width={WIDTH} height={ROW} />
          <text class="label" x={LABEL_W - 8} y={y + BAR / 2 + 4} text-anchor="end">
            {b.label.length > 24 ? `${b.label.slice(0, 23)}…` : b.label}
          </text>
          <path class="bar" d={barPath(w, y)} />
          <text class="value" x={LABEL_W + w + 6} y={y + BAR / 2 + 4}>{text(b.value)}</text>
        </g>
      {/each}
    </svg>
    {#if hover !== null && sorted[hover]}
      {@const b = sorted[hover]}
      <div class="tooltip" style:top="{hover * ROW + ROW + 34}px">
        <b>{text(b.value)}</b>
        <span>{b.label}{b.sub ? ` · ${b.sub}` : ""}</span>
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
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
    margin-bottom: 0.3rem;
    font-size: 0.9rem;
  }
  figcaption .sub {
    font-size: 0.75rem;
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
    font-size: 10px;
    fill: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .label {
    font-size: 11px;
    fill: var(--text);
  }
  .value {
    font-size: 11px;
    fill: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .bar {
    fill: var(--accent);
    transition: opacity 0.1s;
  }
  .hit {
    fill: transparent;
  }
  .row {
    outline: none;
  }
  .row.dim .bar {
    opacity: 0.45;
  }
  .tooltip {
    position: absolute;
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
  .empty {
    margin: 0.5rem 0;
  }
</style>
