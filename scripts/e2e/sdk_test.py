"""End-to-end API compatibility check with the official OpenAI and Anthropic SDKs.

Needs a running contiod whose config defines the aliases `coder` (a tool-capable
chat model) and `embed-gemma` (FLM embeddings), see docs/config.example.toml.

    uv run --no-project --with openai --with anthropic python scripts/e2e/sdk_test.py
    CONTIO_URL=http://127.0.0.1:18080 uv run ...   # other address
"""
import os
import sys

import anthropic
import openai

B = os.environ.get("CONTIO_URL", "http://127.0.0.1:8080")
failures = 0
oa = openai.OpenAI(base_url=B + "/v1", api_key="none")
an = anthropic.Anthropic(base_url=B, api_key="none")
weather = {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}

def check(name, cond, detail=""):
    global failures
    failures += not cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

# --- OpenAI chat
r = oa.chat.completions.create(model="coder", messages=[{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=600, temperature=0)
check("chat via alias", "pong" in (r.choices[0].message.content or "").lower(), repr(r.choices[0].message.content[-30:]))

# --- OpenAI chat streaming with tools (guard rewrite path)
calls = {}
for ch in oa.chat.completions.create(model="coder", messages=[{"role": "user", "content": "What's the weather in Karlsruhe?"}],
                                     tools=[{"type": "function", "function": weather}], max_tokens=2048, temperature=0, stream=True):
    for tc in (ch.choices[0].delta.tool_calls or []) if ch.choices else []:
        calls.setdefault(tc.index, [tc.function.name, ""])[1] += tc.function.arguments or ""
check("chat stream + tools", calls and calls[0][0] == "get_weather" and "Karlsruhe" in calls[0][1], str(calls))

# --- response_format validation report
r = oa.chat.completions.create(model="coder", messages=[{"role": "user", "content": "Person named Ada, 36."}], max_tokens=1500, temperature=0,
    response_format={"type": "json_schema", "json_schema": {"name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}})
ext = r.model_extra.get("contio")
check("response_format validated", ext is not None and ext.get("schema_valid") is True, str(ext))

# --- Anthropic non-streaming tool use
m = an.messages.create(model="coder", max_tokens=2048, tools=[{"name": "get_weather", "description": "weather", "input_schema": weather["parameters"]}],
                       messages=[{"role": "user", "content": "What's the weather in Karlsruhe?"}])
tu = [b for b in m.content if b.type == "tool_use"]
check("anthropic tool_use", m.stop_reason == "tool_use" and tu and tu[0].input.get("city") == "Karlsruhe", f"{m.stop_reason} {[b.type for b in m.content]}")

# --- Anthropic tool result round trip, streaming
msgs = [{"role": "user", "content": "What's the weather in Karlsruhe?"},
        {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in m.content if b.type != "thinking"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0].id, "content": "18°C, sunny"}]}]
with an.messages.stream(model="coder", max_tokens=1500, messages=msgs,
                        tools=[{"name": "get_weather", "description": "weather", "input_schema": weather["parameters"]}]) as s:
    final = s.get_final_message()
text = "".join(b.text for b in final.content if b.type == "text")
check("anthropic stream after tool_result", "18" in text and final.stop_reason == "end_turn", repr(text[:80]))
check("anthropic usage", final.usage.output_tokens > 0, str(final.usage))

# --- Anthropic count_tokens
ct = an.messages.count_tokens(model="coder", messages=[{"role": "user", "content": "hello world"}])
check("anthropic count_tokens", ct.input_tokens > 0, str(ct.input_tokens))

# --- Responses: function call, then previous_response_id follow-up (streaming)
r1 = oa.responses.create(model="coder", input="What's the weather in Karlsruhe?", max_output_tokens=2048,
                         tools=[{"type": "function", **weather}])
fc = [o for o in r1.output if o.type == "function_call"]
check("responses function_call", fc and "Karlsruhe" in fc[0].arguments, f"{r1.status} {[o.type for o in r1.output]}")
events = []
with oa.responses.stream(model="coder", previous_response_id=r1.id, max_output_tokens=1500,
                         input=[{"type": "function_call_output", "call_id": fc[0].call_id, "output": "18°C, sunny"}],
                         tools=[{"type": "function", **weather}]) as s:
    for e in s:
        events.append(e.type)
    r2 = s.get_final_response()
check("responses stream + previous_response_id", "18" in r2.output_text and events[-1] == "response.completed", repr(r2.output_text[:80]))

# --- embeddings via alias on the NPU (co-loaded)
e = oa.embeddings.create(model="embed-gemma", input=["hello"])
check("embeddings alias (NPU)", len(e.data[0].embedding) == 768, str(len(e.data[0].embedding)))

# --- models list
ids = [m.id for m in oa.models.list()]
check("models list incl. aliases", "coder" in ids and "embed-gemma" in ids, str(ids))

sys.exit(1 if failures else 0)
