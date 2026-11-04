<script lang="ts">
  import { page } from "$app/state";
  import { streamChat, type ToolCall, type Usage } from "$lib/chat";
  import { errorText, fetchAliases, fetchModels } from "$lib/daemon";
  import { live } from "$lib/live.svelte";
  import type { Alias, ModelStatus } from "$lib/types";

  interface Stats {
    usage: Usage | null;
    ttft: number | null;
    seconds: number;
    finish: string | null;
  }

  interface Msg {
    role: "user" | "assistant" | "tool";
    content: string;
    /** data: URLs (user messages). */
    images?: string[];
    reasoning?: string;
    tool_calls?: ToolCall[];
    /** Tool results: which call they answer. */
    tool_call_id?: string;
    tool_name?: string;
    stats?: Stats;
    error?: string;
  }

  const EXAMPLE_TOOLS = `[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Current weather for a city",
      "parameters": {
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
      }
    }
  }
]`;

  const saved = (key: string, fallback: string) => localStorage.getItem(`playground.${key}`) ?? fallback;

  let models = $state<ModelStatus[]>([]);
  let aliases = $state<Alias[]>([]);
  let model = $state(page.url.searchParams.get("model") ?? saved("model", ""));
  let system = $state(saved("system", ""));
  let toolsOn = $state(saved("toolsOn", "false") === "true");
  let toolsJson = $state(saved("tools", EXAMPLE_TOOLS));
  let temperature = $state<number | null>(null);
  let maxTokens = $state(2048);
  let showReasoning = $state(true);
  let showRaw = $state(false);

  let messages = $state<Msg[]>([]);
  let input = $state("");
  let images = $state<string[]>([]);
  /** Drafts of tool results, by tool call id. */
  let results = $state<Record<string, string>>({});
  let busy = $state(false);
  let abort: AbortController | null = null;
  let lastRequest = $state<unknown>(null);
  let error = $state<string | null>(null);
  let scroller: HTMLElement | undefined = $state();

  $effect(() => {
    localStorage.setItem("playground.model", model);
    localStorage.setItem("playground.system", system);
    localStorage.setItem("playground.toolsOn", String(toolsOn));
    localStorage.setItem("playground.tools", toolsJson);
  });

  async function loadModels() {
    try {
      const [m, a] = await Promise.all([fetchModels(), fetchAliases()]);
      models = m.filter((x) => x.kind === "chat");
      // upstream_model aliases address another model in the process (FLM's embedder)
      aliases = a.filter((x) => !x.upstream_model && models.some((m) => m.id === x.model));
      if (!model || !names().includes(model)) model = (models.find((x) => x.state === "ready") ?? models[0])?.id ?? "";
    } catch (e) {
      error = errorText(e);
    }
  }
  const names = () => [...models.map((m) => m.id), ...aliases.map((a) => a.name)];

  $effect(() => {
    if (live.online) loadModels();
  });

  let target = $derived(models.find((m) => m.id === (aliases.find((a) => a.name === model)?.model ?? model)));

  let tools = $derived.by((): { ok: true; value: unknown[] } | { ok: false; error: string } => {
    if (!toolsOn) return { ok: true, value: [] };
    try {
      const v = JSON.parse(toolsJson);
      if (!Array.isArray(v)) return { ok: false, error: "tools must be a JSON array" };
      return { ok: true, value: v };
    } catch (e) {
      return { ok: false, error: errorText(e) };
    }
  });

  /** The conversation in OpenAI wire format. */
  function wire(): unknown[] {
    const out: unknown[] = system.trim() ? [{ role: "system", content: system }] : [];
    for (const m of messages) {
      if (m.error) continue;
      if (m.role === "user") {
        out.push({
          role: "user",
          content: m.images?.length
            ? [{ type: "text", text: m.content }, ...m.images.map((url) => ({ type: "image_url", image_url: { url } }))]
            : m.content,
        });
      } else if (m.role === "assistant") {
        const calls = m.tool_calls?.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.arguments } }));
        out.push({ role: "assistant", content: m.content || (calls?.length ? null : ""), ...(calls?.length ? { tool_calls: calls } : {}) });
      } else {
        out.push({ role: "tool", tool_call_id: m.tool_call_id, content: m.content });
      }
    }
    return out;
  }

  async function generate() {
    if (!model || busy) return;
    if (!tools.ok) {
      error = `Tools: ${tools.error}`;
      return;
    }
    error = null;
    const body: Record<string, unknown> = { model, messages: wire(), max_tokens: maxTokens };
    if (temperature != null) body.temperature = temperature;
    if (toolsOn && tools.value.length) body.tools = tools.value;
    lastRequest = body;

    messages.push({ role: "assistant", content: "" });
    const reply = messages[messages.length - 1];
    busy = true;
    abort = new AbortController();
    try {
      const r = await streamChat(
        body,
        (d) => {
          if (d.content) reply.content += d.content;
          if (d.reasoning) reply.reasoning = (reply.reasoning ?? "") + d.reasoning;
          if (d.tool_calls) reply.tool_calls = d.tool_calls;
          scrollDown();
        },
        abort.signal,
      );
      reply.stats = { usage: r.usage, ttft: r.ttft, seconds: r.seconds, finish: r.finish_reason };
    } catch (e) {
      if (abort.signal.aborted) reply.stats = { usage: null, ttft: null, seconds: 0, finish: "stopped" };
      else reply.error = errorText(e);
    } finally {
      busy = false;
      abort = null;
      loadModels();
    }
  }

  function send() {
    const text = input.trim();
    if (!text && !images.length) return;
    messages.push({ role: "user", content: text, images: images.length ? images : undefined });
    input = "";
    images = [];
    scrollDown();
    generate();
  }

  /** Answer all open tool calls of the last assistant turn and continue. */
  function sendResults(m: Msg) {
    for (const c of m.tool_calls ?? []) {
      messages.push({ role: "tool", tool_call_id: c.id, tool_name: c.name, content: results[c.id] ?? "" });
      delete results[c.id];
    }
    generate();
  }

  function onkey(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  async function attach(e: Event) {
    const files = (e.currentTarget as HTMLInputElement).files ?? [];
    for (const f of files) {
      images.push(
        await new Promise<string>((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(String(r.result));
          r.onerror = () => reject(r.error);
          r.readAsDataURL(f);
        }),
      );
    }
    (e.currentTarget as HTMLInputElement).value = "";
  }

  function scrollDown() {
    requestAnimationFrame(() => scroller?.scrollTo({ top: scroller.scrollHeight }));
  }

  function clear() {
    abort?.abort();
    messages = [];
    results = {};
    lastRequest = null;
    error = null;
  }

  function pretty(json: string): { text: string; valid: boolean } {
    try {
      return { text: JSON.stringify(JSON.parse(json), null, 2), valid: true };
    } catch {
      return { text: json, valid: false };
    }
  }

  const tps = (s: Stats) =>
    s.usage && s.ttft != null && s.seconds > s.ttft ? s.usage.completion_tokens / (s.seconds - s.ttft) : null;
  const open = (m: Msg, i: number) =>
    m.role === "assistant" && !!m.tool_calls?.length && i === messages.length - 1 && !busy && !m.error;
</script>

<section class="card settings">
  <div class="row">
    <label>
      Model
      <select bind:value={model}>
        {#each models as m (m.id)}
          <option value={m.id}>{m.id} · {m.device.toUpperCase()}{m.state === "ready" ? " · loaded" : ""}</option>
        {/each}
        {#each aliases as a (a.name)}
          <option value={a.name}>{a.name} → {a.model}</option>
        {/each}
      </select>
    </label>
    {#if target && target.state !== "ready"}<span class="sub">loads on the first message</span>{/if}
    <span class="spacer"></span>
    <label>Temperature <input type="number" min="0" max="2" step="0.1" placeholder="default" bind:value={temperature} /></label>
    <label>Max tokens <input type="number" min="1" step="256" bind:value={maxTokens} /></label>
  </div>
  {#if models.length === 0 && live.online}
    <p class="sub">No chat models installed. Pick one in the <a href="/catalog">catalog</a>.</p>
  {/if}
  <details>
    <summary>System prompt{system.trim() ? " (set)" : ""}</summary>
    <textarea rows="3" bind:value={system} placeholder="You are a helpful assistant."></textarea>
  </details>
  <details open={toolsOn}>
    <summary>Tools{toolsOn && tools.ok ? ` (${tools.value.length})` : ""}</summary>
    <label class="check"><input type="checkbox" bind:checked={toolsOn} /> send these tool definitions</label>
    <textarea class="mono" rows="10" bind:value={toolsJson} spellcheck="false"></textarea>
    {#if toolsOn && !tools.ok}<p class="error-text">{tools.error}</p>{/if}
  </details>
  <div class="row sub">
    <label class="check"><input type="checkbox" bind:checked={showReasoning} /> show reasoning</label>
    <label class="check"><input type="checkbox" bind:checked={showRaw} /> show last request</label>
    <span class="spacer"></span>
    <button class="small" disabled={!messages.length} onclick={clear}>New chat</button>
  </div>
</section>

<section class="card chat" bind:this={scroller}>
  {#if messages.length === 0}
    <p class="sub empty">Send a message to start. Enter sends, Shift+Enter adds a line.</p>
  {/if}
  {#each messages as m, i (i)}
    <div class="msg {m.role}">
      <div class="who">{m.role === "tool" ? `tool result · ${m.tool_name}` : m.role}</div>
      {#if m.images?.length}
        <div class="thumbs">{#each m.images as src, j (j)}<img {src} alt="attachment {j + 1}" />{/each}</div>
      {/if}
      {#if m.reasoning && showReasoning}
        <details class="reasoning" open={busy && i === messages.length - 1 && !m.content}>
          <summary>reasoning</summary>
          <div class="text">{m.reasoning}</div>
        </details>
      {/if}
      {#if m.content}
        <div class="text" class:mono={m.role === "tool"}>{m.content}</div>
      {:else if m.role === "assistant" && busy && i === messages.length - 1 && !m.reasoning && !m.tool_calls}
        <div class="sub">{target?.state === "ready" ? "thinking…" : "loading the model…"}</div>
      {/if}
      {#each m.tool_calls ?? [] as c (c.id || c.name)}
        {@const args = pretty(c.arguments)}
        <div class="call">
          <div class="call-head">
            <b>{c.name}</b>
            <span class="sub mono">{c.id}</span>
            <span class="pill {args.valid ? 'fits' : 'too_big'}">{args.valid ? "valid JSON" : "invalid JSON"}</span>
          </div>
          <pre class="cmd">{args.text}</pre>
          {#if open(m, i)}
            <textarea class="mono" rows="2" placeholder={'result, e.g. {"temp_c": 18}'} bind:value={results[c.id]}></textarea>
          {/if}
        </div>
      {/each}
      {#if open(m, i)}
        <button class="primary small" onclick={() => sendResults(m)}>Send tool results</button>
      {/if}
      {#if m.error}<p class="error-text">{m.error}</p>{/if}
      {#if m.stats}
        {@const t = tps(m.stats)}
        <div class="stats sub">
          {#if m.stats.usage}{m.stats.usage.prompt_tokens} in · {m.stats.usage.completion_tokens} out ·{/if}
          {#if m.stats.ttft != null}first token {m.stats.ttft.toFixed(2)} s ·{/if}
          {#if t != null}{t.toFixed(1)} tok/s ·{/if}
          {m.stats.finish ?? ""}
        </div>
      {/if}
    </div>
  {/each}
</section>

{#if error}<p class="error-text">{error}</p>{/if}

<section class="card composer">
  {#if images.length}
    <div class="thumbs">
      {#each images as src, j (j)}
        <button class="thumb" title="Remove" onclick={() => images.splice(j, 1)}><img {src} alt="attachment {j + 1}" /></button>
      {/each}
    </div>
  {/if}
  <textarea rows="3" placeholder="Message {model || '…'}" bind:value={input} onkeydown={onkey}></textarea>
  <div class="row">
    <label class="attach small">
      Attach image
      <input type="file" accept="image/*" multiple onchange={attach} hidden />
    </label>
    <span class="spacer"></span>
    {#if busy}
      <button onclick={() => abort?.abort()}>Stop</button>
    {:else}
      <button class="primary" disabled={!model || (!input.trim() && !images.length)} onclick={send}>Send</button>
    {/if}
  </div>
</section>

{#if showRaw && lastRequest}
  <section class="card">
    <h2>Last request</h2>
    <pre class="cmd raw">POST /v1/chat/completions (stream)
{JSON.stringify(lastRequest, (k, v) => (typeof v === "string" && v.startsWith("data:") ? `${v.slice(0, 40)}…` : v), 2)}</pre>
  </section>
{/if}

<style>
  .settings {
    display: grid;
    gap: 0.5rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .row label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--muted);
    font-size: 0.9rem;
  }
  .row input[type="number"] {
    width: 6rem;
  }
  .spacer {
    flex: 1;
  }
  details summary {
    cursor: pointer;
    color: var(--muted);
    font-size: 0.9rem;
  }
  details textarea {
    width: 100%;
    box-sizing: border-box;
    margin-top: 0.4rem;
  }
  .check {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    font-size: 0.9rem;
  }
  .mono {
    font-family: ui-monospace, monospace;
    font-size: 0.85rem;
  }
  .chat {
    min-height: 12rem;
    max-height: 60vh;
    overflow: auto;
    display: grid;
    gap: 0.8rem;
    align-content: start;
  }
  .empty {
    margin: 0;
  }
  .msg {
    display: grid;
    gap: 0.35rem;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    background: var(--bg);
  }
  .msg.user {
    background: color-mix(in srgb, var(--accent) 8%, var(--card));
  }
  .msg.tool {
    border-left: 3px solid var(--warn);
  }
  .who {
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .reasoning .text {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.3rem;
  }
  .call {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem;
    background: var(--card);
    display: grid;
    gap: 0.35rem;
  }
  .call-head {
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }
  .call pre {
    margin: 0;
  }
  .call textarea {
    width: 100%;
    box-sizing: border-box;
  }
  .msg > button {
    justify-self: start;
  }
  .stats {
    font-size: 0.8rem;
  }
  .thumbs {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .thumbs img {
    height: 4rem;
    border-radius: 4px;
    border: 1px solid var(--border);
  }
  .thumb {
    padding: 0;
    border: none;
    background: none;
    cursor: pointer;
  }
  .composer {
    display: grid;
    gap: 0.5rem;
  }
  .composer textarea {
    width: 100%;
    box-sizing: border-box;
    resize: vertical;
  }
  .attach {
    cursor: pointer;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
  }
  .raw {
    max-height: 24rem;
    overflow: auto;
  }
</style>
