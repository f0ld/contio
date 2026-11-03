<script lang="ts">
  import { onMount } from "svelte";
  import { errorText, fetchModels, formatBytes, formatContext, loadModel, removeModel, unloadModel } from "$lib/daemon";
  import type { ModelStatus } from "$lib/types";

  let { online }: { online: boolean } = $props();

  let models = $state<ModelStatus[]>([]);
  let busy = $state<Record<string, "loading" | "unloading" | "removing">>({});
  let errors = $state<Record<string, string>>({});
  /** Model whose remove confirmation is open. */
  let confirming = $state<string | null>(null);
  let deleteFiles = $state(false);

  async function refresh() {
    try {
      models = await fetchModels();
    } catch {
      // connection state is shown in the header; keep the last list
    }
  }

  async function act(id: string, kind: "loading" | "unloading" | "removing") {
    busy[id] = kind;
    delete errors[id];
    try {
      if (kind === "loading") await loadModel(id);
      else if (kind === "unloading") await unloadModel(id);
      else {
        await removeModel(id, deleteFiles);
        confirming = null;
      }
    } catch (e) {
      errors[id] = errorText(e);
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
    <p class="sub">No models installed yet. Pick one in the <a href="/catalog">catalog</a>.</p>
  {:else}
    <table class="list">
      <thead>
        <tr><th>Model</th><th>Device</th><th>Context</th><th>Memory</th><th>State</th><th></th></tr>
      </thead>
      <tbody>
        {#each models as m (m.id)}
          {@const action = busy[m.id]}
          <tr>
            <td>
              <span class="id">{m.id}</span>
              {#if m.pinned}<span class="tag accent">pinned</span>{/if}
              {#if m.kind !== "chat"}<span class="tag">{m.kind}</span>{/if}
              <div class="sub">
                {m.backend ?? "no backend"}{m.origin?.quant && m.origin.quant !== "none" ? ` · ${m.origin.quant}` : ""}{m.managed ? "" : " · config.toml"}
              </div>
            </td>
            <td class="upper">{m.device}</td>
            <td class="num">{formatContext(m.context)}</td>
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
              {#if m.managed}
                <button class="danger" title="Remove" disabled={!online || !!action}
                        onclick={() => { confirming = confirming === m.id ? null : m.id; deleteFiles = false; }}>✕</button>
              {/if}
            </td>
          </tr>
          {#if confirming === m.id}
            <tr class="confirm-row">
              <td colspan="6">
                Remove <b>{m.id}</b>?
                <label><input type="checkbox" bind:checked={deleteFiles} /> also delete its files</label>
                <button class="danger" disabled={!!action} onclick={() => act(m.id, "removing")}>Remove</button>
                <button onclick={() => (confirming = null)}>Cancel</button>
              </td>
            </tr>
          {/if}
          {#if errors[m.id] || m.state === "failed"}
            <tr class="error-row">
              <td colspan="6"><pre>{errors[m.id] ?? (m.state === "failed" ? m.error : "")}</pre></td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  .id {
    font-weight: 600;
    margin-right: 0.3rem;
  }
  .upper {
    text-transform: uppercase;
    font-size: 0.85rem;
  }
  .state {
    font-size: 0.85rem;
  }
  .state.ready {
    color: var(--ok);
  }
  .state.loading,
  .state.unloading,
  .state.removing {
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
    white-space: nowrap;
  }
  .actions button {
    font-size: 0.85rem;
    padding: 0.2rem 0.7rem;
  }
  .confirm-row td {
    border-top: none;
    font-size: 0.9rem;
  }
  .confirm-row label {
    margin: 0 0.75rem;
    color: var(--muted);
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
