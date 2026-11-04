<script lang="ts">
  import { streamChat } from "$lib/chat";
  import { castVote, errorText, fetchArena, fetchModels } from "$lib/daemon";
  import { live } from "$lib/live.svelte";
  import type { ArenaRating, ArenaWinner, ModelStatus } from "$lib/types";

  interface Side {
    model: string;
    text: string;
    reasoning: string;
    done: boolean;
    error: string | null;
    seconds: number | null;
  }

  let models = $state<ModelStatus[]>([]);
  let pool = $state<string[]>([]);
  let ratings = $state<ArenaRating[]>([]);
  let votes = $state(0);
  let prompt = $state("");
  let sides = $state<[Side, Side] | null>(null);
  let voted = $state<ArenaWinner | null>(null);
  let error = $state<string | null>(null);
  let asked = $state("");

  async function load() {
    try {
      const [m, a] = await Promise.all([fetchModels(), fetchArena()]);
      models = m.filter((x) => x.kind === "chat");
      if (!pool.length) pool = models.map((x) => x.id);
      ratings = a.ratings;
      votes = a.votes;
    } catch (e) {
      error = errorText(e);
    }
  }

  $effect(() => {
    if (live.online) load();
  });

  const toggle = (id: string) => (pool = pool.includes(id) ? pool.filter((x) => x !== id) : [...pool, id]);

  async function ask() {
    if (pool.length < 2 || !prompt.trim()) return;
    error = null;
    voted = null;
    asked = prompt.trim();
    // two different models, random sides
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    const blank = (model: string): Side => ({ model, text: "", reasoning: "", done: false, error: null, seconds: null });
    sides = [blank(shuffled[0]), blank(shuffled[1])];
    const current = sides;
    await Promise.all(
      current.map(async (side) => {
        try {
          const r = await streamChat({ model: side.model, messages: [{ role: "user", content: asked }], max_tokens: 4096 }, (d) => {
            if (d.content) side.text += d.content;
            if (d.reasoning) side.reasoning += d.reasoning;
          });
          side.seconds = r.seconds;
        } catch (e) {
          side.error = errorText(e);
        } finally {
          side.done = true;
        }
      }),
    );
  }

  async function vote(w: ArenaWinner) {
    if (!sides) return;
    try {
      await castVote({ prompt: asked, model_a: sides[0].model, model_b: sides[1].model, winner: w });
      voted = w;
      await load();
    } catch (e) {
      error = errorText(e);
    }
  }

  let ready = $derived(!!sides && sides.every((s) => s.done));
</script>

<section class="card">
  <div class="head">
    <h2>Arena</h2>
    <span class="sub">the same prompt to two random models; vote before the names are shown</span>
    <span class="spacer"></span>
    <a href="/bench">Benchmarks</a>
  </div>
  <div class="row">
    <span class="label">Models in play</span>
    {#each models as m (m.id)}
      <label class="pick"><input type="checkbox" checked={pool.includes(m.id)} onchange={() => toggle(m.id)} /> {m.id}</label>
    {/each}
  </div>
  {#if pool.length < 2}<p class="sub">Pick at least two models.</p>{/if}
  <textarea rows="3" placeholder="Ask something both models should answer…" bind:value={prompt}
    onkeydown={(e) => e.key === "Enter" && (e.ctrlKey || e.metaKey) && ask()}></textarea>
  <div class="row">
    <span class="sub">Ctrl+Enter sends · both models answer at the same time</span>
    <span class="spacer"></span>
    <button class="primary" disabled={pool.length < 2 || !prompt.trim() || (!!sides && !ready)} onclick={ask}>
      {sides && !ready ? "Answering…" : "Ask both"}
    </button>
  </div>
  {#if error}<p class="error-text">{error}</p>{/if}
</section>

{#if sides}
  <div class="pair">
    {#each sides as s, i (i)}
      <section class="card answer" class:win={voted === (i === 0 ? "a" : "b")}>
        <div class="head">
          <b>{i === 0 ? "A" : "B"}</b>
          {#if voted}<span class="tag accent">{s.model}</span>{/if}
          <span class="spacer"></span>
          {#if s.seconds != null}<span class="sub">{s.seconds.toFixed(1)} s</span>{/if}
        </div>
        {#if s.reasoning}
          <details><summary class="sub">reasoning</summary><div class="text sub">{s.reasoning}</div></details>
        {/if}
        {#if s.error}<p class="error-text">{s.error}</p>{/if}
        <div class="text">{s.text}{#if !s.done && !s.text}<span class="sub">…</span>{/if}</div>
      </section>
    {/each}
  </div>
  {#if ready && !voted}
    <section class="card votes">
      <button onclick={() => vote("a")}>A is better</button>
      <button onclick={() => vote("tie")}>Tie</button>
      <button onclick={() => vote("both_bad")}>Both bad</button>
      <button onclick={() => vote("b")}>B is better</button>
    </section>
  {:else if voted}
    <p class="sub center">Vote saved. A was {sides[0].model}, B was {sides[1].model}.</p>
  {/if}
{/if}

<section class="card">
  <div class="head">
    <h2>Ratings</h2>
    <span class="sub">Elo from {votes} vote{votes === 1 ? "" : "s"}; starts at 1000</span>
  </div>
  {#if ratings.length === 0}
    <p class="sub">No votes yet.</p>
  {:else}
    <table class="list">
      <thead><tr><th>#</th><th>Model</th><th class="num">Elo</th><th class="num">Games</th><th class="num">Won</th><th class="num">Lost</th><th class="num">Tied</th></tr></thead>
      <tbody>
        {#each ratings as r, i (r.model)}
          <tr>
            <td class="sub">{i + 1}</td>
            <td><b>{r.model}</b></td>
            <td class="num">{Math.round(r.elo)}</td>
            <td class="num">{r.games}</td>
            <td class="num">{r.wins}</td>
            <td class="num">{r.losses}</td>
            <td class="num">{r.ties}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  .head,
  .row {
    display: flex;
    align-items: center;
    gap: 0.5rem 0.9rem;
    flex-wrap: wrap;
  }
  .head h2 {
    margin: 0;
  }
  .head {
    margin-bottom: 0.5rem;
  }
  .row {
    margin: 0.5rem 0;
  }
  .label {
    color: var(--muted);
    font-size: 0.85rem;
  }
  .pick {
    font-size: 0.9rem;
    display: flex;
    gap: 0.3rem;
    align-items: center;
  }
  .spacer {
    flex: 1;
  }
  textarea {
    width: 100%;
    box-sizing: border-box;
  }
  .pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .answer {
    margin: 0;
  }
  .answer.win {
    outline: 2px solid var(--ok);
  }
  .text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: 0.95rem;
  }
  .votes {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
  }
  .center {
    text-align: center;
  }
</style>
