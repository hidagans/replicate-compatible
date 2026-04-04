# Replicate Compatible API

Server FastAPI sederhana yang bertindak sebagai layer kompatibilitas untuk model-model di **Replicate**, memungkinkan Anda menggunakan endpoint bergaya **OpenAI** dan **Anthropic** untuk berinteraksi dengan model Replicate.

## Fitur Utama

- **Kompatibilitas OpenAI**: Mendukung endpoint `/v1/chat/completions`.
- **Kompatibilitas Anthropic**: Mendukung endpoint `/v1/messages`.
- **Streaming Support**: Mendukung streaming respons (Server-Sent Events) untuk kedua gaya API.
- **Model Mapping**: Memetakan nama model populer (seperti `gpt-4` atau `gpt-3.5-turbo`) ke model spesifik di Replicate secara otomatis.
- **Autentikasi Fleksibel**: Mendukung `Authorization: Bearer <TOKEN>` (gaya OpenAI) dan `x-api-key: <TOKEN>` (gaya Anthropic) menggunakan Replicate API Token Anda.

## Persyaratan

- Python 3.10+
- Replicate API Token (Dapatkan di [replicate.com/account](https://replicate.com/account))

## Instalasi

1. Clone repositori ini atau download kodenya.
2. Instal dependensi yang diperlukan:

   ```bash
   pip install -r requirements.txt
   ```

3. Buat file `.env` di direktori akar dan tambahkan konfigurasi (opsional):

   ```env
   REPLICATE_MODEL_ID=meta/llama-2-7b-chat
   LOG_LEVEL=INFO
   ```

## Menjalankan Server

Jalankan server menggunakan `uvicorn`:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Server sekarang akan berjalan di `http://localhost:8000`.

## Cara Penggunaan

### 1. Endpoint OpenAI (`/v1/chat/completions`)

Anda bisa menggunakan client OpenAI atau `curl` biasa:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -d '{
    "model": "meta/llama-2-7b-chat",
    "messages": [{"role": "user", "content": "Halo, siapa kamu?"}],
    "stream": true
  }'
```

### 2. Endpoint Anthropic (`/v1/messages`)

Gunakan Replicate API Token Anda sebagai `x-api-key`:

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $REPLICATE_API_TOKEN" \
  -d '{
    "model": "anthropic/claude-3-opus",
    "messages": [{"role": "user", "content": "Apa kabar?"}],
    "max_tokens": 1024
  }'
```

## Pemetaan Model (Model Mapping)

Anda dapat mengubah pemetaan model di dalam [server.py](file:///c:/Users/MAJESTY%20YEARI/Documents/ngoding/replicate-compatible/server.py) pada variabel `MODEL_MAP`. Secara default:

- `gpt-3.5-turbo` -> `meta/llama-2-7b-chat`
- `gpt-4` -> `meta/llama-2-70b-chat`

Jika model yang diminta tidak ada dalam pemetaan dan tidak menyertakan owner (contoh: `llama-3-70b`), server akan mencoba menambahkan owner default dari `REPLICATE_MODEL_ID`.

## Lisensi

MIT
