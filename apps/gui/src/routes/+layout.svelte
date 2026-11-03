<script lang="ts">
  import "../app.css";
  import { onMount } from "svelte";
  import { page } from "$app/state";
  import { DAEMON_URL } from "$lib/daemon";
  import { jobs, startJobPolling } from "$lib/jobs.svelte";
  import { live, startLive } from "$lib/live.svelte";

  let { children } = $props();

  onMount(() => {
    const stopLive = startLive();
    const stopJobs = startJobPolling();
    return () => {
      stopLive();
      stopJobs();
    };
  });

  const links = [
    { href: "/", label: "Dashboard" },
    { href: "/catalog", label: "Catalog" },
    { href: "/jobs", label: "Jobs" },
    { href: "/backends", label: "Backends" },
  ];
  const active = (href: string) =>
    href === "/" ? page.url.pathname === "/" : page.url.pathname.startsWith(href);
</script>

<header>
  <h1>contio</h1>
  <nav>
    {#each links as l (l.href)}
      <a href={l.href} class:active={active(l.href)}>
        {l.label}
        {#if l.href === "/jobs" && jobs.running.length}<span class="badge">{jobs.running.length}</span>{/if}
      </a>
    {/each}
  </nav>
  <span class="conn {live.conn}" title={DAEMON_URL}>
    {live.conn === "open" ? "connected" : live.conn === "connecting" ? "connecting…" : "daemon offline"}
  </span>
</header>

<main>
  {@render children()}
</main>

<style>
  header {
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.6rem 1.25rem;
    background: var(--card);
    border-bottom: 1px solid var(--border);
  }
  h1 {
    margin: 0;
    font-size: 1.25rem;
  }
  nav {
    display: flex;
    gap: 0.25rem;
    flex: 1;
  }
  nav a {
    color: var(--muted);
    text-decoration: none;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
  }
  nav a:hover {
    color: var(--text);
  }
  nav a.active {
    color: var(--text);
    background: var(--bg);
    font-weight: 600;
  }
  .badge {
    display: inline-block;
    min-width: 1.1rem;
    margin-left: 0.25rem;
    padding: 0 0.3rem;
    border-radius: 999px;
    background: var(--accent);
    color: white;
    font-size: 0.75rem;
    text-align: center;
  }
  .conn {
    font-size: 0.85rem;
    color: var(--muted);
    white-space: nowrap;
  }
  .conn::before {
    content: "●";
    margin-right: 0.35rem;
  }
  .conn.open::before {
    color: var(--ok);
  }
  .conn.connecting::before {
    color: var(--warn);
  }
  .conn.closed::before {
    color: var(--err);
  }
  main {
    max-width: 1000px;
    margin: 0 auto;
    padding: 1.25rem;
    display: grid;
    gap: 1rem;
  }
</style>
