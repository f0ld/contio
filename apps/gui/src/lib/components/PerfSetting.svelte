<script lang="ts">
  import type { Snippet } from "svelte";
  import type { PerfSetting } from "$lib/performance";

  /** One expert setting: control, what it does, upsides (+) and downsides (−). */
  let { setting, children, note = null }: { setting: PerfSetting; children: Snippet; note?: string | null } = $props();
</script>

<div class="perf">
  <div class="head">{@render children()} <b>{setting.title}</b></div>
  <p class="sub">{setting.what}{note ? ` ${note}` : ""}</p>
  <ul>
    {#each setting.upsides as u (u)}<li><span class="mark" aria-label="upside">+</span>{u}</li>{/each}
    {#each setting.downsides as d (d)}<li><span class="mark" aria-label="downside">−</span>{d}</li>{/each}
  </ul>
</div>

<style>
  .perf {
    padding: 0.5rem 0;
    border-top: 1px solid var(--border);
    margin-top: 0.4rem;
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
  }
  p {
    margin: 0.2rem 0;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    font-size: 0.85rem;
  }
  .mark {
    display: inline-block;
    width: 1.1rem;
    font-weight: 700;
    color: var(--muted);
  }
</style>
