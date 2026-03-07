import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MODEL_ID = os.getenv("REPLICATE_MODEL_ID", "openai/gpt-5.2")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
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
        payload["messages"] = json.dumps([m.dict() for m in req.messages])
    elif req.prompt is not None:
        payload["prompt"] = req.prompt
    if req.image_input:
        payload["image_input"] = req.image_input
    if req.reasoning_effort:
        payload["reasoning_effort"] = req.reasoning_effort
    if req.verbosity:
        payload["verbosity"] = req.verbosity
    if req.max_tokens:
        payload["max_completion_tokens"] = req.max_tokens
    return payload

def ensure_token():
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN tidak terpasang")

@app.get("/v1/models")
def list_models():
    ensure_token()
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
async def chat_completions(req: ChatRequest, request: Request):
    ensure_token()
    replicate_input = build_replicate_input(req)
    model = req.model or MODEL_ID
    if req.stream:
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
                err = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(err)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        try:
            output = replicate.run(model, input=replicate_input)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        content = output if isinstance(output, str) else str(output)
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
