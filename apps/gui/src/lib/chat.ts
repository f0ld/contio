// Streaming client for the OpenAI chat API served by contiod (`/v1/chat/completions`).

import { DAEMON_URL } from "./daemon";

export interface ToolCall {
  id: string;
  name: string;
  /** Raw JSON string as produced by the model. */
  arguments: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface StreamResult {
  finish_reason: string | null;
  usage: Usage | null;
  /** Seconds from request to the first generated token (content, reasoning or tool call). */
  ttft: number | null;
  seconds: number;
}

export interface Delta {
  content?: string;
  reasoning?: string;
  tool_calls?: ToolCall[];
}

interface ChunkCall {
  index?: number;
  id?: string;
  function?: { name?: string; arguments?: string };
}

/**
 * POST `body` with `stream: true` and call `ondelta` for every chunk. Tool
 * calls are merged by index and reported as the full list so far.
 */
export async function streamChat(
  body: Record<string, unknown>,
  ondelta: (d: Delta) => void,
  signal?: AbortSignal,
): Promise<StreamResult> {
  const started = performance.now();
  const res = await fetch(`${DAEMON_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...body, stream: true, stream_options: { include_usage: true } }),
    signal,
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => null);
    throw new Error(err?.error?.message ?? `HTTP ${res.status}`);
  }

  const calls: ToolCall[] = [];
  const result: StreamResult = { finish_reason: null, usage: null, ttft: null, seconds: 0 };
  const firstToken = () => (result.ttft ??= (performance.now() - started) / 1000);

  const handle = (data: string) => {
    if (data === "[DONE]") return;
    const chunk = JSON.parse(data);
    if (chunk.error) throw new Error(chunk.error.message ?? JSON.stringify(chunk.error));
    if (chunk.usage) result.usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (!choice) return;
    if (choice.finish_reason) result.finish_reason = choice.finish_reason;
    const delta = choice.delta ?? {};
    const out: Delta = {};
    if (delta.content) out.content = delta.content;
    if (delta.reasoning_content) out.reasoning = delta.reasoning_content;
    for (const c of (delta.tool_calls ?? []) as ChunkCall[]) {
      const i = c.index ?? calls.length;
      calls[i] ??= { id: "", name: "", arguments: "" };
      if (c.id) calls[i].id = c.id;
      if (c.function?.name) calls[i].name += c.function.name;
      if (c.function?.arguments) calls[i].arguments += c.function.arguments;
      out.tool_calls = calls.map((x) => ({ ...x }));
    }
    if (Object.keys(out).length) {
      firstToken();
      ondelta(out);
    }
  };

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trimEnd();
      buffer = buffer.slice(nl + 1);
      if (line.startsWith("data:")) handle(line.slice(5).trim());
    }
  }
  result.seconds = (performance.now() - started) / 1000;
  return result;
}
