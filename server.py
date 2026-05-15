import os
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Union
import logging

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("replicate_compat")

app = FastAPI()

MODEL_ID = os.getenv("REPLICATE_MODEL_ID", "openai/gpt-5.2")

MODEL_MAP = {
    "gpt-3.5-turbo": "meta/llama-2-7b-chat",
    "gpt-4": "meta/llama-2-70b-chat",
    "gpt-5.4": "openai/gpt-5.4",
    "deepseek-v3": "deepseek-ai/deepseek-v3",
    "deepseek-ai/deepseek-v3": "deepseek-ai/deepseek-v3",
}

# Models that use a flat prompt string instead of a messages array
_PROMPT_ONLY_MODELS = {
    "deepseek-ai/deepseek-v3",
}


def resolve_model_id(m: Optional[str]) -> str:
    m = m or MODEL_ID
    if m in MODEL_MAP:
        return MODEL_MAP[m]
    if "/" not in m:
        default_owner = MODEL_ID.split("/")[0] if "/" in MODEL_ID else "openai"
        return f"{default_owner}/{m}"
    return m


# ─── Tool Calling ─────────────────────────────────────────────────────────────
#
# Replicate models don't have native function-calling support.
# We emulate it via prompt injection + multi-format response parsing.
#
# Strategy:
#   1. Inject tool schemas into the system prompt with a clear format + examples.
#   2. Parse the model output using 6 different heuristics (see parse_tool_calls).
#   3. Strip parsed blocks before returning content to the caller.

TOOL_SYSTEM_PROMPT = """\
You have access to external tools. When you need to call a tool, output ONLY a \
JSON object in the following format — nothing else on that line:

{{"name": "<tool_name>", "arguments": {{<key>: <value>, ...}}}}

Rules:
- Output one JSON object per tool call, on its own line.
- Do NOT wrap the JSON in markdown code fences.
- Do NOT add prose before or after the JSON when calling a tool.
- After you see the tool result, continue your answer normally.
- If you don't need any tool, answer directly without any JSON.

Available tools (JSON Schema):
{tools_json}

Example — if you need to call "get_weather" with location "Paris":
{{"name": "get_weather", "arguments": {{"location": "Paris"}}}}
"""


def _simplify_tools(tools: List[Dict]) -> List[Dict]:
    simplified = []
    for t in tools:
        if t.get("type") == "function":
            fn = t.get("function", {})
            simplified.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        else:
            simplified.append(t)
    return simplified


def build_tool_system_prompt(tools: List[Dict]) -> str:
    return TOOL_SYSTEM_PROMPT.format(
        tools_json=json.dumps(_simplify_tools(tools), indent=2)
    )


def _try_parse_json(s: str) -> Optional[Dict]:
    """Best-effort JSON parse with light auto-fix."""
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Attempt to fix trailing commas: ,} or ,]
    fixed = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return None


def _make_tool_call(data: Dict) -> Optional[Dict]:
    """
    Normalise a parsed dict into an OpenAI tool-call object.
    Accepts several naming conventions models use in practice:
        {name, arguments}          ← our primary format
        {name, parameters}
        {tool, arguments/parameters}
        {function_name, arguments/parameters}
        {tool_name, tool_input}    ← Anthropic-ish
    """
    # Resolve name
    fn_block = data.get("function")
    name = (
        data.get("name")
        or data.get("tool")
        or data.get("tool_name")
        or data.get("function_name")
        or (fn_block.get("name") if isinstance(fn_block, dict) else None)
    )
    if not name or not isinstance(name, str):
        return None

    # Resolve arguments
    args = (
        data.get("arguments")
        or data.get("parameters")
        or data.get("tool_input")
        or data.get("input")
        or {}
    )
    if isinstance(args, str):
        parsed = _try_parse_json(args)
        args = parsed if parsed is not None else {}
    if not isinstance(args, dict):
        args = {}

    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


# Regex that matches a JSON object with up to 2 levels of brace nesting.
# Covers: {"name": "fn", "arguments": {"key": "val"}}
# Pattern breakdown: \{ ... (\{ ... \} | non-brace)* ... \}
_JSON_OBJ = r'\{(?:[^{}]|\{[^{}]*\})*\}'

# Ordered list of extraction strategies (tried in sequence, first wins)
_TOOL_CALL_PATTERNS: List[tuple] = [
    # 1. <tool_call>...</tool_call> (our legacy format + Anthropic style)
    ("xml_tool_call",      re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>",       re.DOTALL)),
    # 2. <function_call>...</function_call>
    ("xml_function_call",  re.compile(r"<function_call>\s*(.*?)\s*</function_call>", re.DOTALL)),
    # 3. [TOOL_CALL] ... [/TOOL_CALL]
    ("bracket_tool_call",  re.compile(r"\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]",   re.DOTALL)),
    # 4. ```json ... ``` / ``` ... ``` code block
    ("json_codeblock",     re.compile(r"```(?:json)?\s*(" + _JSON_OBJ + r")\s*```", re.DOTALL)),
    # 5. Bare JSON object on its own line (handles nested braces e.g. "arguments": {...})
    ("bare_json_line",     re.compile(r"^\s*(" + _JSON_OBJ + r")\s*$",              re.MULTILINE)),
]


def parse_tool_calls(text: str) -> List[Dict]:
    """
    Extract tool calls from model output using multiple format heuristics.
    Returns a list of OpenAI-format tool_call objects.
    """
    results: List[Dict] = []
    seen_names: set = set()

    for strategy, pattern in _TOOL_CALL_PATTERNS:
        for raw in pattern.findall(text):
            raw = raw.strip()
            data = _try_parse_json(raw)
            if data is None:
                continue
            tc = _make_tool_call(data)
            if tc is None:
                continue
            # Dedup by function name (same tool called once per response)
            key = tc["function"]["name"]
            if key not in seen_names:
                seen_names.add(key)
                results.append(tc)
                logger.debug(f"Tool call detected via '{strategy}': {tc['function']['name']}")

    return results


def strip_tool_calls(text: str) -> str:
    """Remove all recognised tool-call blocks from the response text."""
    # XML-style tags
    text = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<function_call>\s*.*?\s*</function_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[TOOL_CALL\]\s*.*?\s*\[/TOOL_CALL\]", "", text, flags=re.DOTALL)
    # JSON code blocks
    text = re.sub(r"```(?:json)?\s*" + _JSON_OBJ + r"\s*```", "", text, flags=re.DOTALL)
    # Bare JSON lines that look like tool calls (handles nested braces)
    text = re.sub(r"^\s*" + _JSON_OBJ + r"\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_tool_result_message(content: str, name: Optional[str], tool_call_id: Optional[str]) -> str:
    """Format a tool result so the model understands it's a tool response."""
    label = name or tool_call_id or "tool"
    return f"[Tool result from '{label}']:\n{content}"


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[Union[str, List[ChatMessage]]] = None
    system: Optional[Any] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    image_input: Optional[List[str]] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    functions: Optional[List[Dict]] = None  # legacy OpenAI format
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


# ─── Build Replicate Input ────────────────────────────────────────────────────

def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("type")
                if t in ("text", "input_text"):
                    parts.append(part.get("text") or part.get("input_text") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def _messages_to_prompt(messages_list: List[Dict], system_prompt: str) -> str:
    """
    Convert a messages array into a single prompt string.
    Used for models like deepseek-ai/deepseek-v3 that only accept a flat prompt.
    Format: <system>\n{system}\n</system>\n\nUser: ...\n\nAssistant: ...
    """
    parts = []
    if system_prompt:
        parts.append(f"<system>\n{system_prompt}\n</system>")
    for m in messages_list:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        # system already handled above; tool/other roles become User lines
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def build_replicate_input(req: ChatRequest, model_name: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    is_anthropic = model_name.startswith("anthropic/")
    is_prompt_only = model_name in _PROMPT_ONLY_MODELS

    # Normalize tools (support both tools[] and legacy functions[])
    tools = req.tools or []
    if not tools and req.functions:
        tools = [{"type": "function", "function": f} for f in req.functions]

    # ── Build message list ──────────────────────────────────────────────────
    messages_list = []
    if req.messages:
        raw = req.messages
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raise HTTPException(status_code=400, detail="messages must be valid JSON array")

        for m in (raw if isinstance(raw, list) else []):
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = _flatten_content(m.get("content", ""))
                tool_calls = m.get("tool_calls")
                tool_call_id = m.get("tool_call_id")
                name = m.get("name")
            else:
                role = m.role
                content = _flatten_content(m.content)
                tool_calls = m.tool_calls
                tool_call_id = m.tool_call_id
                name = m.name

            # Tool result → user message with clear label
            if role == "tool":
                content = _format_tool_result_message(content, name, tool_call_id)
                role = "user"

            # Assistant with tool_calls → embed call in content so model has context
            if role == "assistant" and tool_calls:
                tc_lines = []
                for tc in tool_calls:
                    if tc.get("type") != "function":
                        continue
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    try:
                        fn_args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        fn_args = {}
                    tc_lines.append(json.dumps({"name": fn_name, "arguments": fn_args}))
                if tc_lines:
                    tc_block = "\n".join(tc_lines)
                    content = f"{content}\n{tc_block}".strip() if content else tc_block

            messages_list.append({"role": role, "content": content})

    # ── Build system prompt ─────────────────────────────────────────────────
    system_parts = []
    if req.system:
        system_parts.append(_flatten_content(req.system) if not isinstance(req.system, str) else req.system)

    # Pull system messages out of the message list
    filtered_messages = []
    for m in messages_list:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            filtered_messages.append(m)
    messages_list = filtered_messages

    # Append tool schema to system prompt
    if tools:
        system_parts.append(build_tool_system_prompt(tools))

    system_prompt = "\n\n".join(p for p in system_parts if p).strip()

    # ── Assemble provider-specific payload ─────────────────────────────────
    if is_anthropic:
        if system_prompt:
            payload["system"] = system_prompt
        if messages_list:
            payload["messages"] = json.dumps(messages_list)
        elif req.prompt:
            payload["prompt"] = req.prompt
        if req.max_tokens:
            tokens = int(req.max_tokens)
            tokens = max(1024, tokens) if "claude-4.5" in model_name else max(16, tokens)
            payload["max_tokens"] = tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p

    elif is_prompt_only:
        # Models like deepseek-ai/deepseek-v3 only accept a flat prompt string.
        # Convert messages → prompt, or pass through req.prompt directly.
        if messages_list:
            payload["prompt"] = _messages_to_prompt(messages_list, system_prompt)
        elif req.prompt is not None:
            prompt_text = req.prompt
            if system_prompt:
                prompt_text = f"<system>\n{system_prompt}\n</system>\n\n{prompt_text}"
            payload["prompt"] = prompt_text
        if req.max_tokens is not None:
            payload["max_tokens"] = max(1, int(req.max_tokens))
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.presence_penalty is not None:
            payload["presence_penalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            payload["frequency_penalty"] = req.frequency_penalty

    else:
        # OpenAI style
        if system_prompt:
            messages_list = [{"role": "system", "content": system_prompt}] + messages_list
        if messages_list:
            payload["messages"] = messages_list
        elif req.prompt is not None:
            payload["prompt"] = req.prompt
        if req.max_tokens is not None:
            tokens = max(16, int(req.max_tokens))
            payload["max_output_tokens"] = tokens
            payload["max_completion_tokens"] = tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.presence_penalty is not None:
            payload["presence_penalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            payload["frequency_penalty"] = req.frequency_penalty

    if req.image_input:
        payload["image_input"] = req.image_input
    if req.reasoning_effort:
        payload["reasoning_effort"] = req.reasoning_effort
    if req.verbosity:
        payload["verbosity"] = req.verbosity

    return payload


# ─── Auth ─────────────────────────────────────────────────────────────────────

def get_replicate_token(request: Request):
    auth = request.headers.get("Authorization")
    api_token = None
    if auth and auth.startswith("Bearer "):
        api_token = auth.split(" ", 1)[1].strip()
    if not api_token:
        api_token = request.headers.get("x-api-key")
    if not api_token:
        api_token = request.headers.get("anthropic-key") or request.headers.get("api-key")
    if not api_token:
        raise HTTPException(status_code=401, detail="Unauthorized: No Replicate token in headers")
    os.environ["REPLICATE_API_TOKEN"] = api_token
    return api_token


# ─── Helper: build model list ─────────────────────────────────────────────────

def _build_model_list() -> list:
    now = int(time.time())
    models = []
    primary_id = resolve_model_id(None)
    models.append(
        {
            "id": primary_id,
            "object": "model",
            "created": now,
            "owned_by": "replicate",
            "context_length": 128000,
            "max_context_length": 128000,
            "max_completion_tokens": 16384,
            "max_tokens": 16384,
        }
    )
    seen = {primary_id}
    for alias, replicate_id in MODEL_MAP.items():
        for mid in (alias, replicate_id):
            if mid not in seen:
                seen.add(mid)
                models.append(
                    {
                        "id": mid,
                        "object": "model",
                        "created": now,
                        "owned_by": "replicate",
                        "context_length": 128000,
                        "max_context_length": 128000,
                        "max_completion_tokens": 16384,
                        "max_tokens": 16384,
                    }
                )
    return models


# ─── Routes ───────────────────────────────────────────────────────────────────

# ── OpenAI model listing ──────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models_v1(_: Any = Depends(get_replicate_token)):
    return JSONResponse({"object": "list", "data": _build_model_list()})


@app.get("/v1/models/{model_id:path}")
def get_model_v1(model_id: str, _: Any = Depends(get_replicate_token)):
    now = int(time.time())
    resolved = resolve_model_id(model_id)
    return JSONResponse(
        {
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "replicate",
            "context_length": 128000,
            "max_context_length": 128000,
            "max_completion_tokens": 16384,
            "max_tokens": 16384,
            "replicate_model": resolved,
        }
    )


# ── Ollama compat ─────────────────────────────────────────────────────────────

@app.get("/api/tags")
def ollama_tags():
    return JSONResponse({"server": "replicate-compatible", "tags": []})


# ── LM Studio compat ──────────────────────────────────────────────────────────

@app.get("/api/v1/models")
def lmstudio_models(_: Any = Depends(get_replicate_token)):
    return JSONResponse({"server": "replicate-compatible", "models": []})


# ── llama.cpp compat ──────────────────────────────────────────────────────────

@app.get("/v1/props")
def llama_props_v1():
    return JSONResponse({"server": "replicate-compatible", "version": "1.0.0"})


@app.get("/props")
def llama_props():
    return JSONResponse({"server": "replicate-compatible", "version": "1.0.0"})


# ── vLLM / generic version ────────────────────────────────────────────────────

@app.get("/version")
def server_version():
    return JSONResponse({"server": "replicate-compatible", "server_version": "1.0.0"})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Chat completions ──────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatRequest, request: Request, _: Any = Depends(get_replicate_token)
):
    logger.info("POST /v1/chat/completions model=%s stream=%s", req.model, req.stream)
    if not req.messages and (req.prompt is None or req.prompt == ""):
        raise HTTPException(status_code=400, detail="Provide messages or prompt")

    model = resolve_model_id(req.model)
    if "/" not in model:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model}")

    has_tools = bool(req.tools or req.functions)
    replicate_input = build_replicate_input(req, model)
    logger.debug("Replicate input: %s", json.dumps(replicate_input)[:800])

    rid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # ── Streaming ──────────────────────────────────────────────────────────
    if req.stream:
        def event_generator():
            try:
                # ── Collect all chunks from Replicate ─────────────────────
                # replicate.stream() has two behaviors depending on model:
                #   Incremental: each chunk = new token only  (OpenAI-style)
                #   Cumulative : each chunk = full text so far (DeepSeek etc.)
                #
                # Cumulative mode is unreliable to detect mid-stream because
                # chunks can go backward in length between iterations.
                # Safest strategy: collect ALL chunks, then deduplicate.
                raw_chunks: List[str] = []
                for chunk in replicate.stream(model, input=replicate_input):
                    raw_chunks.append(chunk if isinstance(chunk, str) else str(chunk))

                # ── Detect cumulative vs incremental stream ───────────────
                # Cumulative : each chunk = full response so far (DeepSeek etc.)
                #              → final content = LAST chunk
                # Incremental: each chunk = new tokens only (OpenAI-style)
                #              → final content = JOIN all chunks
                #
                # Detection strategy:
                #   In cumulative mode, chunk[i] is always a prefix of chunk[i+1].
                #   We sample pairs from the MIDDLE of the stream (not the start,
                #   where chunks can be very short and cause false positives).
                #   Require the prefix chunk to be ≥ 10 chars to count.
                if raw_chunks:
                    n = len(raw_chunks)
                    is_cumulative = False

                    if n >= 3:
                        # Skip the first couple of chunks (too short, unreliable)
                        start = min(2, n - 2)
                        sample_end = min(start + 5, n - 1)
                        pairs = [(raw_chunks[i], raw_chunks[i + 1]) for i in range(start, sample_end)]
                        # Only count pairs where the prefix chunk is meaningful (≥10 chars)
                        valid_pairs = [(a, b) for a, b in pairs if len(a) >= 10]
                        if valid_pairs:
                            matches = sum(1 for a, b in valid_pairs if b.startswith(a))
                            is_cumulative = matches >= max(1, len(valid_pairs) * 0.6)

                    if is_cumulative:
                        content_acc = raw_chunks[-1]
                        logger.debug("Cumulative stream mode: using last chunk (%d chars)", len(content_acc))
                    else:
                        content_acc = "".join(raw_chunks)
                        logger.debug("Incremental stream mode: joined %d chunks (%d chars)", len(raw_chunks), len(content_acc))
                else:
                    content_acc = ""

                # ── Post-process tool calls ────────────────────────────────
                tool_calls = parse_tool_calls(content_acc) if has_tools else []
                clean_content = strip_tool_calls(content_acc) if tool_calls else content_acc

                # ── Re-stream to client as proper SSE deltas ───────────────
                # Emit the full content as a single delta so the client
                # never sees partial/repeated content.
                if clean_content:
                    data = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": clean_content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tc_chunk = {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "index": i,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": tc["function"]["arguments"],
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tc_chunk)}\n\n"

                    fin = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    }
                    yield f"data: {json.dumps(fin)}\n\n"
                else:
                    fin = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(fin)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Streaming error")
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ── Non-streaming ──────────────────────────────────────────────────────
    try:
        output = replicate.run(model, input=replicate_input)
    except Exception as e:
        logger.exception("Replicate run error")
        msg = str(e)
        lower = msg.lower()
        status = 500
        if "unauthorized" in lower or "401" in lower:
            status = 401
        elif "bad request" in lower or "400" in lower or "422" in lower:
            status = 400
        elif "rate limit" in lower or "429" in lower:
            status = 429
        raise HTTPException(status_code=status, detail=msg)

    content = output if isinstance(output, str) else str(output)
    logger.debug("Raw model output (first 500): %s", content[:500])

    tool_calls = parse_tool_calls(content) if has_tools else []
    clean_content = strip_tool_calls(content) if tool_calls else content
    finish_reason = "tool_calls" if tool_calls else "stop"

    if tool_calls:
        logger.info("Detected %d tool call(s): %s", len(tool_calls), [tc["function"]["name"] for tc in tool_calls])

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": clean_content if clean_content else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    resp = {
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }
    return JSONResponse(resp)


# ── Anthropic messages ────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def anthropic_messages(
    req: ChatRequest, request: Request, _: Any = Depends(get_replicate_token)
):
    logger.info("POST /v1/messages (Anthropic style)")
    model = resolve_model_id(req.model)
    if "/" not in model:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model}")

    replicate_input = build_replicate_input(req, model)

    if req.stream:
        def event_generator():
            msg_id = f"msg_{uuid.uuid4().hex}"
            try:
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                for chunk in replicate.stream(model, input=replicate_input):
                    token = chunk if isinstance(chunk, str) else str(chunk)
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            except Exception as e:
                logger.exception("Anthropic streaming error")
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            output = replicate.run(model, input=replicate_input)
            content = output if isinstance(output, str) else str(output)
            return JSONResponse(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                    "model": model,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            )
        except Exception as e:
            logger.exception("Anthropic run error")
            raise HTTPException(status_code=500, detail=str(e))


# ─── Middleware & Exception Handlers ─────────────────────────────────────────

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    try:
        body = await request.body()
        if body:
            logger.debug("Raw body: %s", body[:2000].decode("utf-8", errors="ignore"))
    except Exception:
        pass
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_text = body[:2000].decode("utf-8", errors="ignore") if body else ""
    except Exception:
        body_text = ""
    logger.error("Validation error: %s | body: %s", exc.errors(), body_text)
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )
