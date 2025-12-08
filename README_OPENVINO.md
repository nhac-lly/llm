# OpenVINO GenAI - Clean Implementation

Clean, production-ready implementation using **OpenVINO GenAI** for NPU/GPU/CPU inference.

## Features

- ✅ **Native OpenVINO support** - Uses `openvino-genai` (not ONNX Runtime)
- ✅ **NPU/GPU/CPU** - Automatic device selection with priority: NPU > GPU > CPU
- ✅ **OpenVINO IR format** - Works with `.xml/.bin` model files
- ✅ **Streaming support** - Real-time token generation
- ✅ **REST API** - Flask API with Swagger documentation
- ✅ **CLI tool** - Command-line interface for quick testing

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

Or install manually:
```powershell
pip install openvino-genai openvino huggingface-hub flask flask-restx
```

### 2. Download/Convert a Model

**Option A: Download pre-converted OpenVINO model**
```powershell
python agent_openvino.py --download microsoft/Phi-3-mini-4k-instruct
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
python agent_openvino.py "models/phi3-openvino" "Write a haiku"
python agent_openvino.py "models/phi3-openvino" --interactive
```

**API:**
```powershell
python api_openvino.py --model "models/phi3-openvino"
# Visit http://127.0.0.1:5000/docs
```

## Files

- **`agent_openvino.py`** - CLI tool (download, generate, interactive)
- **`api_openvino.py`** - REST API with Swagger UI
- **`download_openvino_model.py`** - Model conversion helper
- **`requirements.txt`** - Dependencies

## CLI Usage

### Download Model
```powershell
python agent_openvino.py --download microsoft/Phi-3-mini-4k-instruct
```

### Generate Text
```powershell
python agent_openvino.py "path/to/model" "Your prompt here"
```

### Interactive Mode
```powershell
python agent_openvino.py "path/to/model" --interactive
```

### Device Selection
```powershell
# Auto-select (NPU > GPU > CPU)
python agent_openvino.py "path/to/model" "prompt" --device auto

# Force NPU
python agent_openvino.py "path/to/model" "prompt" --device npu

# Force GPU
python agent_openvino.py "path/to/model" "prompt" --device gpu

# Force CPU
python agent_openvino.py "path/to/model" "prompt" --device cpu
```

## API Usage

### Start Server
```powershell
python api_openvino.py --model "path/to/model" --device npu
```

### Swagger UI
Visit: **http://127.0.0.1:5000/docs**

### Endpoints
- `GET /api/health` - Health check
- `POST /api/model/load` - Load model
- `POST /api/model/unload` - Unload model
- `POST /api/llm/generate` - Generate text
- `POST /api/llm/chat` - Stream generation
- `GET /api/llm/stop` - Stop generation

### Example Request
```bash
# Load model
curl -X POST http://127.0.0.1:5000/api/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/phi3-openvino", "device": "npu"}'

# Generate text
curl -X POST http://127.0.0.1:5000/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku", "max_length": 100}'
```

## Model Format

OpenVINO GenAI requires models in **OpenVINO IR format** (`.xml` + `.bin` files), not ONNX.

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
```

## Differences from ONNX Runtime Version

| Feature | ONNX Runtime GenAI | OpenVINO GenAI |
|---------|-------------------|----------------|
| Model Format | ONNX (.onnx) | OpenVINO IR (.xml/.bin) |
| NPU Support | Limited | Native |
| Library | `onnxruntime-genai` | `openvino-genai` |
| Device Priority | Manual | Automatic fallback |

## References

- [Intel AI Playground](https://github.com/intel/AI-Playground/tree/main/OpenVINO)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino-genai)

