<script lang="ts">
  import PerfSetting from "$lib/components/PerfSetting.svelte";
  import { errorText, fetchBackends, fetchSettings, installBackend, saveSettings } from "$lib/daemon";
  import { jobs, refreshJobs } from "$lib/jobs.svelte";
  import { PERF_SETTINGS } from "$lib/performance";
  import type { BackendsInfo, Settings } from "$lib/types";
  import { daemonInfo, daemonStart, daemonStop, isDesktop, serviceSet, type DaemonInfo } from "$lib/desktop";
  import { live } from "$lib/live.svelte";

  const desktop = isDesktop();
  let info = $state<DaemonInfo | null>(null);
  let busy = $state<string | null>(null);
  let error = $state<string | null>(null);

  async function refresh() {
    try {
      info = await daemonInfo();
    } catch (e) {
      error = errorText(e);
    }
  }

  async function act(label: string, f: () => Promise<DaemonInfo>) {
    busy = label;
    error = null;
    try {
      info = await f();
    } catch (e) {
      error = errorText(e);
      await refresh();
    } finally {
      busy = null;
    }
  }

  $effect(() => {
    // re-check whenever the connection changes (started or stopped elsewhere)
    void live.online;
    if (desktop) refresh();
  });

  let settings = $state<Settings | null>(null);
  $effect(() => {
    if (live.online && !settings) fetchSettings().then((s) => (settings = s), (e) => (error = errorText(e)));
  });

  async function setServePower(v: Settings["serve_power"]) {
    if (!settings) return;
    try {
      settings = await saveSettings({ ...settings, serve_power: v });
    } catch (e) {
      error = errorText(e);
    }
  }

  // Experimental llama.cpp build (temporary; remove with LLAMACPP_VULKAN_FA_PREVIEW in contio-install)
  let preview = $state<BackendsInfo["preview"] | null>(null);
  let previewJob = $state<number | null>(null);
  let previewBusy = $state(false);
  $effect(() => {
    if (live.online && !preview) fetchBackends().then((b) => (preview = b.preview), (e) => (error = errorText(e)));
  });
  // reload when the build finishes
  $effect(() => {
    if (jobs.byId(previewJob)?.state === "done") fetchBackends().then((b) => (preview = b.preview));
  });

  async function buildPreview() {
    if (!preview) return;
    try {
      previewJob = (await installBackend("llama.cpp", "source", "vulkan", preview.version)).job_id;
      refreshJobs();
    } catch (e) {
      error = errorText(e);
    }
  }

  async function setPreview(on: boolean) {
    if (!settings) return;
    previewBusy = true;
    try {
      settings = await saveSettings({ ...settings, vulkan_fa_preview: on });
    } catch (e) {
      error = errorText(e);
    } finally {
      previewBusy = false;
    }
  }

  const OWNER = {
    app: "started by this app; it stops when you close the window",
    service: "running as the background service",
    external: "started outside this app (e.g. from a terminal)",
    none: "",
  } as const;
</script>

<section class="card">
  <h2>Background service</h2>
  {#if !desktop}
    <p class="sub">Daemon controls are available in the desktop app. Here, start <code>contiod</code> yourself.</p>
  {:else if !info}
    <p class="sub">Checking…</p>
  {:else}
    <div class="row">
      <span class="pill {info.running ? 'fits' : 'tight'}">{info.running ? "running" : "stopped"}</span>
      <span class="sub">{OWNER[info.owner]}</span>
      <span class="spacer"></span>
      {#if info.running}
        <button disabled={!!busy} onclick={() => act("stop", daemonStop)}>{busy === "stop" ? "Stopping…" : "Stop"}</button>
      {:else}
        <button class="primary" disabled={!!busy || (!info.bundled && !info.service_installed)} onclick={() => act("start", daemonStart)}>
          {busy === "start" ? "Starting…" : "Start"}
        </button>
      {/if}
    </div>

    <label class="toggle">
      <input
        type="checkbox"
        checked={info.service_installed}
        disabled={!!busy || !info.service_available || (!info.bundled && !info.service_installed)}
        onchange={(e) => act("service", () => serviceSet(e.currentTarget.checked))}
      />
      <span>
        <b>Run in the background</b>
        <span class="sub">
          Keeps the API available without this window and starts it when you log in (a systemd user service,
          <code>contiod.service</code>). The daemon is copied to <code>~/.local/share/contio/bin</code> and updated
          when you start a newer version of the app.
        </span>
      </span>
    </label>
    {#if busy === "service"}<p class="sub">Setting up the service…</p>{/if}
    {#if info.service_installed && info.service_version}<p class="sub">Service: {info.service_version}</p>{/if}
    {#if !info.service_available}<p class="sub">systemd user services aren't available on this system.</p>{/if}
    {#if !info.bundled && !info.service_installed}
      <p class="sub">This build doesn't include the daemon (development build): run <code>contiod</code> yourself.</p>
    {/if}
  {/if}
  {#if error}<p class="error-text">{error}</p>{/if}
</section>

{#if settings}
  <section class="card">
    <h2><span class="badge">Expert</span> Performance</h2>
    <p class="sub">
      Settings for the whole daemon. Per-model settings are in each model's editor. Measured on this laptop in
      <code>docs/PERFORMANCE.md</code>.
    </p>
    <PerfSetting setting={PERF_SETTINGS.serve_power}>
      <select value={settings.serve_power} onchange={(e) => setServePower(e.currentTarget.value as Settings["serve_power"])}>
        <option value="off">Off</option>
        <option value="on_ac">Only on AC power</option>
        <option value="always">Always</option>
      </select>
    </PerfSetting>
  </section>
{/if}

{#if settings && preview}
  {@const job = jobs.byId(previewJob)}
  <section class="card experimental">
    <h2><span class="badge">Experimental</span><span class="badge">Temporary</span> Faster long prompts on Vulkan</h2>
    <p class="sub">
      Runs Vulkan models on a llama.cpp build of an <b>unmerged</b> upstream change,
      <a href={preview.url} target="_blank" rel="noreferrer">pull request #31323</a> (taller flash-attention tiles on
      Radeon iGPUs), built from source at a fixed commit (<code>{preview.version}</code>). It is not a llama.cpp
      release: it may have bugs, and this option will be removed once the change is in a release.
    </p>
    <ul class="sub facts">
      <li>Measured on the Radeon 890M: prompt reading <b>+29% at 32k</b> and <b>+60% at 64k</b> tokens of context; writing unchanged.</li>
      <li>Only models on Vulkan use it. ROCm models are unaffected.</li>
      <li>Switching restarts loaded Vulkan models on their next request. Switch off to go back to the regular build.</li>
    </ul>

    <div class="actions">
      <button disabled={job?.state === "running"} onclick={buildPreview}>
        {preview.installed ? "Rebuild" : "Build it"}
      </button>
      {#if job}
        <span class="sub">
          job #{job.id}: {job.state === "running" ? job.step ?? "starting" : job.state} · <a href="/jobs">details</a>
        </span>
      {:else}
        <span class="sub">
          {preview.installed
            ? "Built."
            : "~5 minutes; needs the same packages as building llama.cpp from source (Backends → Check prerequisites)."}
        </span>
      {/if}
    </div>
    {#if job?.error}<p class="error-text">{job.error}</p>{/if}

    <label class="toggle">
      <input
        type="checkbox"
        checked={settings.vulkan_fa_preview}
        disabled={previewBusy || (!preview.installed && !settings.vulkan_fa_preview)}
        onchange={(e) => setPreview(e.currentTarget.checked)}
      />
      <span>
        <b>Use the experimental build for Vulkan models</b>
        <span class="sub">{preview.installed ? "" : "Build it first."}</span>
      </span>
    </label>
  </section>
{/if}

<style>
  .badge {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border: 1px solid var(--warn);
    color: var(--warn);
    border-radius: 4px;
    padding: 0 0.3rem;
    margin-right: 0.3rem;
    vertical-align: middle;
  }
  .experimental {
    border-color: var(--warn);
  }
  .facts {
    margin: 0.5rem 0 0.75rem;
    padding-left: 1.1rem;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0 1rem;
  }
  .spacer {
    flex: 1;
  }
  .toggle {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
  }
  .toggle input {
    margin-top: 0.25rem;
  }
  .toggle span {
    display: grid;
    gap: 0.2rem;
  }
  p {
    margin: 0.5rem 0 0;
  }
</style>
