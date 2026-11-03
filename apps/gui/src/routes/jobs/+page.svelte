<script lang="ts">
  import { cancelJob, errorText, formatBytes, retryJob } from "$lib/daemon";
  import { jobs, refreshJobs } from "$lib/jobs.svelte";
  import type { Job } from "$lib/types";

  let open = $state<Record<number, boolean>>({});
  let errors = $state<Record<number, string>>({});

  async function act(job: Job, what: "cancel" | "retry") {
    delete errors[job.id];
    try {
      if (what === "cancel") await cancelJob(job.id);
      else await retryJob(job.id);
      await refreshJobs();
    } catch (e) {
      errors[job.id] = errorText(e);
    }
  }

  const pct = (j: Job) => (j.progress?.total ? (100 * j.progress.done) / j.progress.total : null);
  const when = (secs: number) => new Date(secs * 1000).toLocaleString();
  function took(j: Job): string {
    const s = (j.finished_at ?? Date.now() / 1000) - j.started_at;
    return s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  }
</script>

<section class="card">
  <h2>Jobs</h2>
  {#if jobs.list.length === 0}
    <p class="sub">No jobs yet. Installs from the <a href="/catalog">catalog</a> and backend setups show up here.</p>
  {/if}
  {#each jobs.list as j (j.id)}
    {@const p = pct(j)}
    <div class="job">
      <div class="head">
        <span class="state {j.state}">{j.state}</span>
        <b>{j.title}</b>
        <span class="sub">#{j.id} · {j.kind} · {when(j.started_at)} · {took(j)}</span>
        <span class="spacer"></span>
        {#if j.state === "running"}
          <button class="small" onclick={() => act(j, "cancel")}>Cancel</button>
        {:else if j.spec && j.retried_as == null && j.state !== "done"}
          <button class="small" onclick={() => act(j, "retry")}>Retry</button>
        {/if}
        {#if j.log.length}
          <button class="small" onclick={() => (open[j.id] = !open[j.id])}>{open[j.id] ? "Hide log" : "Log"}</button>
        {/if}
      </div>
      {#if j.state === "running"}
        <div class="step sub">
          {j.step ?? "starting"}
          {#if j.progress}
            · {formatBytes(j.progress.done)}{j.progress.total ? ` of ${formatBytes(j.progress.total)}` : ""}
          {/if}
        </div>
        {#if p != null}<div class="progress"><div style:width="{p}%"></div></div>{/if}
      {/if}
      {#if j.state === "done" && j.result?.model}
        <div class="sub">installed as <code>{String(j.result.model)}</code></div>
      {/if}
      {#if j.error}
        <div class="error-text">{j.error}{j.retried_as != null ? ` → retried as #${j.retried_as}` : ""}</div>
      {/if}
      {#if errors[j.id]}<div class="error-text">{errors[j.id]}</div>{/if}
      {#if open[j.id]}<pre class="cmd log">{j.log.join("\n")}</pre>{/if}
    </div>
  {/each}
</section>

<style>
  .job {
    padding: 0.6rem 0;
    border-top: 1px solid var(--border);
    display: grid;
    gap: 0.3rem;
  }
  .head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .spacer {
    flex: 1;
  }
  .state {
    font-size: 0.75rem;
    text-transform: uppercase;
    font-weight: 600;
    min-width: 5.5rem;
  }
  .state.running {
    color: var(--accent);
  }
  .state.done {
    color: var(--ok);
  }
  .state.failed {
    color: var(--err);
  }
  .state.cancelled {
    color: var(--muted);
  }
  .log {
    max-height: 16rem;
    overflow: auto;
  }
</style>
