# OpenVINO GenAI

Clean, production-ready implementation using **OpenVINO GenAI** for NPU/GPU/CPU inference.

## Features

- ✅ **Native OpenVINO support** - Uses `openvino-genai` for optimal performance
- ✅ **NPU/GPU/CPU** - Automatic device selection with priority: NPU > GPU > CPU
- ✅ **OpenVINO IR format** - Works with `.xml/.bin` model files
- ✅ **Streaming support** - Real-time token generation
- ✅ **REST API** - Flask API with Swagger documentation
- ✅ **CLI tool** - Command-line interface for quick testing

## Quick Start

### 1. Setup

```powershell
.\setup.ps1
```

Or manually:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download/Convert a Model

**Option A: Download pre-converted OpenVINO model**
```powershell
python agent.py --download <repo_id>
```

**Option B: Convert from Hugging Face**
```powershell
# Install conversion tools
pip install optimum-intel nncf

# Convert model
python download_openvino_model.py microsoft/Phi-3-mini-4k-instruct
```

### 3. Use CLI or API

**CLI:**
```powershell
python agent.py "path/to/model" "Write a haiku"
python agent.py "path/to/model" --interactive
```

**API:**
```powershell
python api.py --model "path/to/model"
# Visit http://127.0.0.1:5000/docs
```

## Files

- **`agent.py`** - CLI tool (download, generate, interactive)
- **`api.py`** - REST API with Swagger UI
- **`download_openvino_model.py`** - Model conversion helper
- **`requirements.txt`** - Dependencies

## CLI Usage

### Download Model
```powershell
python agent.py --download microsoft/Phi-3-mini-4k-instruct
```

### Generate Text
```powershell
python agent.py "path/to/model" "Your prompt here"
```

### Interactive Mode
```powershell
python agent.py "path/to/model" --interactive
```

### Device Selection
```powershell
# Auto-select (NPU > GPU > CPU)
python agent.py "path/to/model" "prompt" --device auto

# Force NPU
python agent.py "path/to/model" "prompt" --device npu

# Force GPU
python agent.py "path/to/model" "prompt" --device gpu

# Force CPU
python agent.py "path/to/model" "prompt" --device cpu
```

## API Usage

### Start Server
```powershell
# Models auto-download from Hugging Face on first run
python api.py

# Or specify device
python api.py --device npu
```

**Models used:**
- **Text Generation**: [OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov](https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov)
- **Embeddings**: [OpenVINO/Qwen3-Embedding-0.6B-int8-ov](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov)

### Swagger UI
Visit: **http://127.0.0.1:5000/docs**

### Endpoints

- `GET /api/health` - Health check (shows model status)
- `POST /api/llm/generate` - Generate text (non-streaming)
- `POST /api/llm/chat` - Stream generation (Server-Sent Events)
- `POST /api/llm/stop` - Stop generation
- `POST /api/embed/` - Generate embedding from text

### Example Requests
```bash
# Generate text
curl -X POST http://127.0.0.1:5000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku", "max_length": 100}'

# Stream generation
curl -X POST http://127.0.0.1:5000/api/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_length": 200}'

# Generate embedding
curl -X POST http://127.0.0.1:5000/api/embed/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "task_description": "Given a web search query, retrieve relevant passages"}'
```

## Model Format

OpenVINO GenAI requires models in **OpenVINO IR format** (`.xml` + `.bin` files).

### Converting Models

Use `optimum-cli` to convert models:

```powershell
pip install optimum-intel nncf

optimum-cli export openvino \
  -m microsoft/Phi-3-mini-4k-instruct \
  --weight-format int4 \
  --sym \
  --group-size -1 \
  --ratio 1.0 \
  ./models/phi3-openvino
```

Or use the helper script:
```powershell
python download_openvino_model.py microsoft/Phi-3-mini-4k-instruct
```

## Device Priority

When `device='auto'` (default):
1. **NPU** - Intel Neural Processing Unit (if available)
2. **GPU** - Intel GPU (if available)
3. **CPU** - Fallback

OpenVINO automatically falls back if a device is unavailable.

## Requirements

```
openvino-genai>=2025.0.0
openvino>=2025.0.0
numpy>=1.26.0
huggingface-hub>=0.20.0
flask>=3.0.0
flask-restx>=1.3.0
optimum-intel>=1.26.0
transformers>=4.40.0
torch>=2.0.0
```

## References

- [Intel AI Playground](https://github.com/intel/AI-Playground/tree/main/OpenVINO)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino-genai)
