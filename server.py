import os
import json
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

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[Union[str, List[ChatMessage]]] = None
    system: Optional[Any] = None  # Anthropic style, can be string or list of parts
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    image_input: Optional[List[str]] = None

def build_replicate_input(req: ChatRequest, model_name: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    is_anthropic = model_name.startswith("anthropic/")
    
    messages_list = []
    if req.messages:
        if isinstance(req.messages, str):
            try:
                parsed = json.loads(req.messages)
            except Exception:
                raise HTTPException(status_code=400, detail="messages harus berupa array JSON valid")
            if not isinstance(parsed, list):
                raise HTTPException(status_code=400, detail="messages harus berupa array")
            
            for m in parsed:
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("type")
                            if t in ("text", "input_text"):
                                parts.append(part.get("text") or part.get("input_text") or "")
                            elif t == "image_url":
                                continue
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "\n".join([p for p in parts if p])
                elif not isinstance(content, str):
                    content = str(content)
                messages_list.append({"role": role, "content": content})
        else:
            for m in req.messages:
                content = m.content
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("type")
                            if t in ("text", "input_text"):
                                parts.append(part.get("text") or part.get("input_text") or "")
                            elif t == "image_url":
                                continue
                        elif isinstance(part, str):
                            parts.append(part)
                    content = "\n".join([p for p in parts if p])
                elif not isinstance(content, str):
                    content = str(content)
                messages_list.append({"role": m.role, "content": content})

    if is_anthropic:
        # Anthropic style: system prompt is separate
        system_msg = ""
        if req.system:
            if isinstance(req.system, list):
                parts = []
                for part in req.system:
                    if isinstance(part, dict):
                        t = part.get("type")
                        if t in ("text", "input_text"):
                            parts.append(part.get("text") or part.get("input_text") or "")
                    elif isinstance(part, str):
                        parts.append(part)
                system_msg = "\n".join([p for p in parts if p])
            else:
                system_msg = str(req.system)

        final_messages = []
        for m in messages_list:
            if m["role"] == "system":
                system_msg += ("\n" + m["content"])
            else:
                final_messages.append(m)
        
        if system_msg:
            payload["system"] = system_msg.strip()
        
        # Some Anthropic models on Replicate (like claude-4.5-sonnet) require a 'prompt' field
        # We'll build a prompt from messages as a fallback or primary field
        prompt_parts = []
        if system_msg:
            prompt_parts.append(f"System: {system_msg.strip()}")
        
        for m in final_messages:
            role_label = "Human" if m["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {m['content']}")
        
        if prompt_parts:
            # Format as a conversation
            payload["prompt"] = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
        if final_messages:
            # Also keep messages for models that support it
            payload["messages"] = json.dumps(final_messages) if not isinstance(final_messages, str) else final_messages
        elif req.prompt:
            payload["prompt"] = req.prompt
            
        if req.max_tokens:
            tokens = int(req.max_tokens)
            # Claude 4.5 Sonnet on Replicate requires min 1024
            if "claude-4.5" in model_name:
                tokens = max(1024, tokens)
            else:
                tokens = max(16, tokens)
            payload["max_tokens"] = tokens
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.top_p is not None:
            payload["top_p"] = req.top_p
    else:
        # Default/OpenAI style on Replicate
        if messages_list:
            payload["messages"] = messages_list
        elif req.prompt is not None:
            payload["prompt"] = req.prompt
            
        if req.max_tokens is not None:
            tokens = max(16, int(req.max_tokens))
            payload["max_output_tokens"] = tokens
            payload["max_completion_tokens"] = tokens
            
    if req.image_input:
        payload["image_input"] = req.image_input
    if req.reasoning_effort:
        payload["reasoning_effort"] = req.reasoning_effort
    if req.verbosity:
        payload["verbosity"] = req.verbosity
        
    return payload

def get_replicate_token(request: Request):
    # Support multiple auth headers
    # 1. OpenAI/standard Bearer token
    auth = request.headers.get("Authorization")
    api_token = None
    
    if auth and auth.startswith("Bearer "):
        api_token = auth.split(" ", 1)[1].strip()
    
    # 2. Anthropic style x-api-key
    if not api_token:
        api_token = request.headers.get("x-api-key")
        
    # 3. Fallback to anthropic-key or api-key
    if not api_token:
        api_token = request.headers.get("anthropic-key") or request.headers.get("api-key")
        
    if not api_token:
        logger.warning(f"Authorization missing. Headers: {dict(request.headers)}")
        raise HTTPException(status_code=401, detail="Unauthorized: No Replicate token found in headers (Authorization: Bearer, x-api-key, etc)")
    
    logger.info("Replicate token provided via headers")
    os.environ["REPLICATE_API_TOKEN"] = api_token
    return api_token

@app.get("/v1/models")
def list_models(_: Any = Depends(get_replicate_token)):
    logger.info("GET /v1/models")
    data = {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "replicate",
            }
        ],
    }
    return JSONResponse(data)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, _: Any = Depends(get_replicate_token)):
    logger.info("POST /v1/chat/completions")
    logger.debug(f"Request body: {json.dumps({k: v for k, v in req.dict().items() if v is not None})}")
    if not req.messages and (req.prompt is None or req.prompt == ""):
        logger.warning("Invalid input: no messages or prompt")
        raise HTTPException(status_code=400, detail="Harus menyediakan messages atau prompt")
    model = req.model or MODEL_ID
    replicate_input = build_replicate_input(req, model)
    logger.debug(f"Replicate input: {json.dumps(replicate_input)}")
    if req.stream:
        logger.info("Streaming response enabled")
        def event_generator():
            rid = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            index = 0
            content_acc = ""
            try:
                for chunk in replicate.stream(model, input=replicate_input):
                    if isinstance(chunk, str):
                        token = chunk
                    else:
                        token = str(chunk)
                    content_acc += token
                    data = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": index,
                                "delta": {"role": "assistant", "content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                end_data = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": index,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(end_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("Streaming error")
                status = 500
                msg = str(e)
                lower = msg.lower()
                if "unauthorized" in lower or "401" in lower:
                    status = 401
                elif "bad request" in lower or "422" in lower:
                    status = 400
                elif "rate limit" in lower or "429" in lower:
                    status = 429
                err = {"error": {"message": msg, "status": status}}
                yield f"data: {json.dumps(err)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        logger.info("Non-stream response")
        try:
            output = replicate.run(model, input=replicate_input)
        except Exception as e:
            logger.exception("Replicate run error")
            msg = str(e)
            lower = msg.lower()
            status = 500
            if "unauthorized" in lower or " 401 " in lower or "status\":401" in lower:
                status = 401
            elif "bad request" in lower or " 400 " in lower or "invalid_request_error" in lower or "422" in lower:
                status = 400
            elif "rate limit" in lower or "429" in lower:
                status = 429
            raise HTTPException(status_code=status, detail=msg)
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        content = output if isinstance(output, str) else str(output)
        logger.debug(f"Output type: {type(output).__name__}")
        logger.debug(f"Output length: {len(content) if isinstance(content, str) else 'n/a'}")
        resp = {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
        return JSONResponse(resp)

@app.post("/v1/messages")
async def anthropic_messages(req: ChatRequest, request: Request, _: Any = Depends(get_replicate_token)):
    logger.info("POST /v1/messages (Anthropic style)")
    model = req.model or MODEL_ID
    replicate_input = build_replicate_input(req, model)
    
    if req.stream:
        def event_generator():
            msg_id = f"msg_{uuid.uuid4().hex}"
            # Anthropic stream format is complex, we provide a simplified version that most clients can parse
            # or map the Replicate stream to Anthropic events
            try:
                # 1. message_start
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                
                # 2. content_block_start
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                
                for chunk in replicate.stream(model, input=replicate_input):
                    token = chunk if isinstance(chunk, str) else str(chunk)
                    # 3. content_block_delta
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"
                
                # 4. content_block_stop
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                
                # 5. message_delta
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                
                # 6. message_stop
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                
            except Exception as e:
                logger.exception("Anthropic streaming error")
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            output = replicate.run(model, input=replicate_input)
            content = output if isinstance(output, str) else str(output)
            
            resp = {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
            return JSONResponse(resp)
        except Exception as e:
            logger.exception("Anthropic run error")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

# Middleware untuk logging raw request body
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    try:
        body = await request.body()
        if body:
            snippet = body[:2000]
            try:
                logger.debug(f"Raw body: {snippet.decode('utf-8', errors='ignore')}")
            except Exception:
                logger.debug(f"Raw body bytes length: {len(snippet)}")
    except Exception:
        logger.debug("Tidak bisa membaca raw body")
    response = await call_next(request)
    return response

# Handler untuk validation error (422) dari FastAPI/Pydantic
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_text = body[:2000].decode("utf-8", errors="ignore") if body else ""
    except Exception:
        body_text = ""
    logger.error(f"Validation error: {exc.errors()} | body: {body_text}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )
