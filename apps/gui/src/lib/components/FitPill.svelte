<script lang="ts">
  import { formatBytes } from "$lib/daemon";
  import type { Fit } from "$lib/types";

  let { fit, label = "", detail = false }: { fit: Fit | null; label?: string; detail?: boolean } = $props();

  const text = { fits: "fits", tight: "tight", too_big: "too big" } as const;
</script>

{#if fit}
  <span
    class="pill {fit.verdict}"
    title="needs ~{formatBytes(fit.needed_bytes)} of {formatBytes(fit.budget_bytes)} usable"
  >
    {label ? `${label} ` : ""}{text[fit.verdict]}{detail ? ` · ${formatBytes(fit.needed_bytes)}` : ""}
  </span>
{:else}
  <span class="sub">–</span>
{/if}
