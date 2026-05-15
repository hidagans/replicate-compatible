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
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replicate_compat")

app = FastAPI()

MODEL_ID = os.getenv("REPLICATE_MODEL_ID", "openai/gpt-5.2")

MODEL_MAP = {
    "gpt-3.5-turbo": "meta/llama-2-7b-chat",
    "gpt-4": "meta/llama-2-70b-chat",
    "gpt-5.4": "openai/gpt-5.4",
}

def resolve_model_id(m: Optional[str]) -> str:
    m = m or MODEL_ID
    if m in MODEL_MAP:
        return MODEL_MAP[m]
    if "/" not in m:
        default_owner = MODEL_ID.split("/")[0] if "/" in MODEL_ID else "openai"
        return f"{default_owner}/{m}"
    return m


# ─── Tool Schema → System Prompt ─────────────────────────────────────────────

TOOL_SYSTEM_PROMPT = """You have access to the following tools. To use a tool, respond with a JSON object wrapped in <tool_call> tags like this:

<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

After the tool result is provided, continue your response. You can call multiple tools if needed.

Available tools:
{tools_json}

Important:
- Always use <tool_call> tags when you want to invoke a tool
- Do NOT say "I can't run commands" — use the terminal tool instead
- For bash commands, use the terminal tool with a "command" argument
"""

def build_tool_system_prompt(tools: List[Dict]) -> str:
    simplified = []
    for t in tools:
        if t.get("type") == "function":
            fn = t.get("function", {})
            simplified.append({
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        else:
            simplified.append(t)
    return TOOL_SYSTEM_PROMPT.format(tools_json=json.dumps(simplified, indent=2))


def parse_tool_calls(text: str):
    """Extract <tool_call>...</tool_call> blocks from model response."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    tool_calls = []
    for m in matches:
        try:
            data = json.loads(m)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data.get("name", ""),
                    "arguments": json.dumps(data.get("arguments", {})),
                }
            })
        except Exception:
            logger.warning(f"Failed to parse tool_call block: {m}")
    return tool_calls


def strip_tool_calls(text: str) -> str:
    """Remove <tool_call> blocks from response text."""
    return re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", text, flags=re.DOTALL).strip()


# ─── Models ──────────────────────────────────────────────────────────────────

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
    # Tool calling support
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    functions: Optional[List[Dict]] = None  # legacy OpenAI format


# ─── Build Replicate Input ────────────────────────────────────────────────────

def build_replicate_input(req: ChatRequest, model_name: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    is_anthropic = model_name.startswith("anthropic/")

    # Normalize tools (support both tools[] and legacy functions[])
    tools = req.tools or []
    if not tools and req.functions:
        tools = [{"type": "function", "function": f} for f in req.functions]

    # Build messages list
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
                content = m.get("content", "")
                tool_calls = m.get("tool_calls")
                tool_call_id = m.get("tool_call_id")
                name = m.get("name")
            else:
                role = m.role
                content = m.content
                tool_calls = m.tool_calls
                tool_call_id = m.tool_call_id
                name = m.name

            # Flatten content if it's a list
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("type")
                        if t in ("text", "input_text"):
                            parts.append(part.get("text") or part.get("input_text") or "")
                    elif isinstance(part, str):
                        parts.append(part)
                content = "\n".join(p for p in parts if p)
            elif content is None:
                content = ""
            else:
                content = str(content)

            # Convert tool result messages to plain text
            if role == "tool":
                content = f"[Tool result for {name or tool_call_id}]: {content}"
                role = "user"

            # Convert assistant tool_calls to readable text so model sees context
            if role == "assistant" and tool_calls:
                tc_text = "\n".join(
                    f"<tool_call>{json.dumps({'name': tc['function']['name'], 'arguments': json.loads(tc['function'].get('arguments', '{}'))})}</tool_call>"
                    for tc in tool_calls if tc.get("type") == "function"
                )
                content = (content + "\n" + tc_text).strip() if content else tc_text

            messages_list.append({"role": role, "content": content})

    # Build system prompt (inject tool schemas if tools present)
    system_parts = []
    if req.system:
        if isinstance(req.system, list):
            for part in req.system:
                if isinstance(part, dict):
                    t = part.get("type")
                    if t in ("text", "input_text"):
                        system_parts.append(part.get("text") or "")
                elif isinstance(part, str):
                    system_parts.append(part)
        else:
            system_parts.append(str(req.system))

    # Extract system message from messages
    filtered_messages = []
    for m in messages_list:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            filtered_messages.append(m)
    messages_list = filtered_messages

    if tools:
        system_parts.append(build_tool_system_prompt(tools))

    system_prompt = "\n\n".join(p for p in system_parts if p).strip()

    # Build payload
    if is_anthropic:
        if system_prompt:
            payload["system"] = system_prompt

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        for m in messages_list:
            label = "Human" if m["role"] == "user" else "Assistant"
            prompt_parts.append(f"{label}: {m['content']}")
        if prompt_parts:
            payload["prompt"] = "\n\n".join(prompt_parts) + "\n\nAssistant:"
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
    # Primary model
    primary_id = resolve_model_id(None)
    models.append({
        "id": primary_id,
        "object": "model",
        "created": now,
        "owned_by": "replicate",
        "context_length": 128000,
        "max_context_length": 128000,
        "max_completion_tokens": 16384,
        "max_tokens": 16384,
    })
    # Mapped aliases — expose both the alias AND the resolved replicate model ID
    seen = {primary_id}
    for alias, replicate_id in MODEL_MAP.items():
        for mid in (alias, replicate_id):
            if mid not in seen:
                seen.add(mid)
                models.append({
                    "id": mid,
                    "object": "model",
                    "created": now,
                    "owned_by": "replicate",
                    "context_length": 128000,
                    "max_context_length": 128000,
                    "max_completion_tokens": 16384,
                    "max_tokens": 16384,
                })
    return models


# ─── Routes ───────────────────────────────────────────────────────────────────

# ── OpenAI-compatible model listing ──────────────────────────────────────────

@app.get("/v1/models")
def list_models_v1(_: Any = Depends(get_replicate_token)):
    """OpenAI-style GET /v1/models — used by Hermes for model validation."""
    return JSONResponse({"object": "list", "data": _build_model_list()})


@app.get("/v1/models/{model_id:path}")
def get_model_v1(model_id: str, _: Any = Depends(get_replicate_token)):
    """OpenAI-style GET /v1/models/{id} — used by Hermes to probe individual models."""
    now = int(time.time())
    resolved = resolve_model_id(model_id)
    return JSONResponse({
        "id": model_id,
        "object": "model",
        "created": now,
        "owned_by": "replicate",
        "context_length": 128000,
        "max_context_length": 128000,
        "max_completion_tokens": 16384,
        "max_tokens": 16384,
        "replicate_model": resolved,
    })


# ── Ollama-compatible endpoints ───────────────────────────────────────────────

@app.get("/api/tags")
def ollama_tags():
    """Ollama GET /api/tags — Hermes probes this to detect Ollama servers."""
    # Return empty list so Hermes doesn't mis-detect this as Ollama
    # (Ollama returns {"models": [...]}, non-Ollama servers return 404/other)
    # We return a valid but empty response so the 404 error goes away,
    # but the "models" key being absent means Hermes won't think it's Ollama.
    return JSONResponse({"server": "replicate-compatible", "tags": []})


# ── LM Studio-compatible endpoints ───────────────────────────────────────────

@app.get("/api/v1/models")
def lmstudio_models(_: Any = Depends(get_replicate_token)):
    """LM Studio native GET /api/v1/models — Hermes probes this to detect LM Studio."""
    # Return a response without the "models" key structure that LM Studio uses,
    # so Hermes won't mis-detect this as LM Studio. Just 200 OK to avoid 404 noise.
    now = int(time.time())
    return JSONResponse({
        "server": "replicate-compatible",
        "models": [],  # empty — prevents mis-detection as LM Studio
    })


# ── llama.cpp-compatible endpoints ───────────────────────────────────────────

@app.get("/v1/props")
def llama_props_v1():
    """llama.cpp GET /v1/props — Hermes probes this to detect llama.cpp servers."""
    # Do NOT include "default_generation_settings" — that's the key Hermes
    # checks for. Return something benign so we're not mis-detected as llamacpp.
    return JSONResponse({
        "server": "replicate-compatible",
        "version": "1.0.0",
    })


@app.get("/props")
def llama_props():
    """llama.cpp fallback GET /props (older builds)."""
    return JSONResponse({
        "server": "replicate-compatible",
        "version": "1.0.0",
    })


# ── vLLM / generic version endpoint ──────────────────────────────────────────

@app.get("/version")
def server_version():
    """Generic GET /version — vLLM style. Hermes probes this to detect vLLM."""
    # Do NOT include a bare "version" key with a semver string alone,
    # because Hermes checks `if "version" in data` to classify as vLLM.
    # Return it nested so detection fails gracefully (no mis-classification).
    return JSONResponse({
        "server": "replicate-compatible",
        "server_version": "1.0.0",
    })


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main chat endpoints ───────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, _: Any = Depends(get_replicate_token)):
    logger.info("POST /v1/chat/completions")
    if not req.messages and (req.prompt is None or req.prompt == ""):
        raise HTTPException(status_code=400, detail="Provide messages or prompt")

    model = resolve_model_id(req.model)
    if "/" not in model:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model}")

    has_tools = bool(req.tools or req.functions)
    replicate_input = build_replicate_input(req, model)
    logger.debug(f"Replicate input: {json.dumps(replicate_input)}")

    rid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if req.stream:
        def event_generator():
            content_acc = ""
            try:
                for chunk in replicate.stream(model, input=replicate_input):
                    token = chunk if isinstance(chunk, str) else str(chunk)
                    content_acc += token
                    data = {
                        "id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": token}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                # After full response, check for tool calls
                tool_calls = parse_tool_calls(content_acc) if has_tools else []
                clean_content = strip_tool_calls(content_acc) if tool_calls else content_acc

                if tool_calls:
                    tc_data = {
                        "id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": clean_content, "tool_calls": tool_calls},
                            "finish_reason": "tool_calls",
                        }],
                    }
                    yield f"data: {json.dumps(tc_data)}\n\n"
                else:
                    end_data = {
                        "id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(end_data)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Streaming error")
                yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
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
        tool_calls = parse_tool_calls(content) if has_tools else []
        clean_content = strip_tool_calls(content) if tool_calls else content
        finish_reason = "tool_calls" if tool_calls else "stop"

        resp = {
            "id": rid, "object": "chat.completion",
            "created": created, "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": clean_content,
                    **({" tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        return JSONResponse(resp)


@app.post("/v1/messages")
async def anthropic_messages(req: ChatRequest, request: Request, _: Any = Depends(get_replicate_token)):
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
            return JSONResponse({
                "id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "model": model, "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })
        except Exception as e:
            logger.exception("Anthropic run error")
            raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    try:
        body = await request.body()
        if body:
            snippet = body[:2000]
            logger.debug(f"Raw body: {snippet.decode('utf-8', errors='ignore')}")
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
    logger.error(f"Validation error: {exc.errors()} | body: {body_text}")
    return JSONResponse(status_code=422, content={"error": {"message": "Request validation failed", "details": exc.errors()}})
