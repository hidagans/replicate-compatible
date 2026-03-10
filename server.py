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
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[Union[str, List[ChatMessage]]] = None
    prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    image_input: Optional[List[str]] = None

def build_replicate_input(req: ChatRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if req.messages:
        if isinstance(req.messages, str):
            try:
                parsed = json.loads(req.messages)
            except Exception:
                raise HTTPException(status_code=400, detail="messages harus berupa array JSON valid")
            if not isinstance(parsed, list):
                raise HTTPException(status_code=400, detail="messages harus berupa array")
            payload["messages"] = parsed
        else:
            payload["messages"] = [m.dict() for m in req.messages]
    elif req.prompt is not None:
        payload["prompt"] = req.prompt
    if req.image_input:
        payload["image_input"] = req.image_input
    if req.reasoning_effort:
        payload["reasoning_effort"] = req.reasoning_effort
    if req.verbosity:
        payload["verbosity"] = req.verbosity
    if req.max_tokens is not None:
        tokens = max(16, int(req.max_tokens))
        payload["max_output_tokens"] = tokens
        payload["max_completion_tokens"] = tokens
    return payload

def get_replicate_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        logger.warning("Authorization header missing or invalid format")
        raise HTTPException(status_code=401, detail="Unauthorized")
    api_token = auth.split(" ", 1)[1].strip()
    if not api_token:
        logger.warning("Authorization token empty")
        raise HTTPException(status_code=401, detail="Unauthorized")
    logger.info("Replicate token provided")
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
    replicate_input = build_replicate_input(req)
    logger.debug(f"Replicate input: {json.dumps(replicate_input)}")
    model = req.model or MODEL_ID
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
