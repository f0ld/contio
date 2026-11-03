<script lang="ts">
  import { onMount } from "svelte";
  import { fetchModels, formatBytes, loadModel, unloadModel } from "$lib/daemon";
  import type { ModelStatus } from "$lib/types";

  let { online }: { online: boolean } = $props();

  let models = $state<ModelStatus[]>([]);
  let busy = $state<Record<string, "loading" | "unloading">>({});
  let errors = $state<Record<string, string>>({});

  async function refresh() {
    try {
      models = await fetchModels();
    } catch {
      // connection state is shown in the header; keep the last list
    }
  }

  async function act(id: string, kind: "loading" | "unloading") {
    busy[id] = kind;
    delete errors[id];
    try {
      if (kind === "loading") await loadModel(id);
      else await unloadModel(id);
    } catch (e) {
      errors[id] = e instanceof Error ? e.message : String(e);
    } finally {
      delete busy[id];
      await refresh();
    }
  }

  onMount(() => {
    refresh();
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  });

  $effect(() => {
    if (online) refresh();
  });

  const stateLabel = (m: ModelStatus) =>
    m.state === "ready" ? (m.in_flight > 0 ? `serving ${m.in_flight}` : "ready") : m.state;
</script>

<section class="card">
  <h2>Models</h2>
  {#if models.length === 0}
    <p class="sub">
      No models configured. Add <code>[[models]]</code> to <code>~/.config/contio/config.toml</code>
      (see <code>docs/config.example.toml</code>) and restart <code>contiod</code>.
    </p>
  {:else}
    <table>
      <thead>
        <tr><th>Model</th><th>Device</th><th>Memory</th><th>State</th><th></th></tr>
      </thead>
      <tbody>
        {#each models as m (m.id)}
          {@const action = busy[m.id]}
          <tr>
            <td>
              <span class="id">{m.id}</span>
              {#if m.pinned}<span class="tag">pinned</span>{/if}
              <div class="sub">{m.backend ?? "no backend"}</div>
            </td>
            <td class="upper">{m.device}</td>
            <td class="num">{m.estimated_bytes != null ? formatBytes(m.estimated_bytes) : "–"}</td>
            <td>
              <span class="state {action ?? m.state}">{action ? `${action}…` : stateLabel(m)}</span>
            </td>
            <td class="actions">
              {#if m.state === "ready"}
                <button disabled={!online || !!action} onclick={() => act(m.id, "unloading")}>Unload</button>
              {:else}
                <button disabled={!online || !!action || m.state === "loading" || !m.backend}
                        onclick={() => act(m.id, "loading")}>Load</button>
              {/if}
            </td>
          </tr>
          {#if errors[m.id] || m.state === "failed"}
            <tr class="error-row">
              <td colspan="5"><pre>{errors[m.id] ?? (m.state === "failed" ? m.error : "")}</pre></td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th {
    text-align: left;
    font-weight: 500;
    color: var(--muted);
    font-size: 0.8rem;
    padding: 0 0.5rem 0.4rem 0;
  }
  td {
    padding: 0.45rem 0.5rem 0.45rem 0;
    border-top: 1px solid var(--border);
    vertical-align: top;
  }
  .id {
    font-weight: 600;
  }
  .tag {
    margin-left: 0.4rem;
    font-size: 0.75rem;
    color: var(--accent);
  }
  .upper {
    text-transform: uppercase;
    font-size: 0.85rem;
  }
  .num {
    font-variant-numeric: tabular-nums;
  }
  .state {
    font-size: 0.85rem;
  }
  .state.ready {
    color: var(--ok);
  }
  .state.loading,
  .state.unloading {
    color: var(--warn);
  }
  .state.failed {
    color: var(--err);
  }
  .state.stopped {
    color: var(--muted);
  }
  .actions {
    text-align: right;
  }
  button {
    font: inherit;
    font-size: 0.85rem;
    padding: 0.25rem 0.8rem;
  }
  .error-row td {
    border-top: none;
    padding-top: 0;
  }
  pre {
    margin: 0;
    font-size: 0.75rem;
    color: var(--err);
    white-space: pre-wrap;
    max-height: 8rem;
    overflow: auto;
  }
</style>
