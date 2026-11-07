<script lang="ts">
  import PerfSetting from "$lib/components/PerfSetting.svelte";
  import { errorText, fetchSettings, saveSettings } from "$lib/daemon";
  import { PERF_SETTINGS } from "$lib/performance";
  import type { Settings } from "$lib/types";
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
