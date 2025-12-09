"""
OpenVINO GenAI REST API
Clean Flask API using OpenVINO GenAI for NPU/GPU/CPU inference.
"""

from flask import Flask, request, Response, jsonify
from flask_restx import Api, Resource, fields
from pathlib import Path
from typing import Iterator, Optional
import json
import argparse
import sys
import os

try:
    import openvino_genai as ov_genai
    OPENVINO_GENAI_AVAILABLE = True
except ImportError:
    OPENVINO_GENAI_AVAILABLE = False
    print("⚠ openvino-genai not installed. Install with: pip install openvino-genai")

try:
    import huggingface_hub as hf_hub
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("⚠ huggingface_hub not installed. Install with: pip install huggingface-hub")

try:
    from optimum.intel import OVModelForFeatureExtraction
    from transformers import AutoTokenizer
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    print("⚠ optimum-intel not installed. Embedding model will not work. Install with: pip install optimum-intel transformers torch")

app = Flask(__name__)
api = Api(app, version='1.0', title='OpenVINO GenAI API', doc='/docs')

# Namespaces
ns_health = api.namespace('health', description='Health check')
ns_llm = api.namespace('llm', description='Text generation')
ns_embed = api.namespace('embed', description='Text embeddings')

api.add_namespace(ns_health, path='/health')
api.add_namespace(ns_llm, path='/llm')
api.add_namespace(ns_embed, path='/embed')

# Hardcoded models - auto-downloads from Hugging Face if needed
TEXT_MODEL_ID = "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"
TEXT_MODEL_PATH = "models/Phi-3.5-mini-instruct-int4-cw-ov"

EMBEDDING_MODEL_ID = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"


class ONNXBackend:
    """OpenVINO GenAI backend."""
    
    def __init__(self):
        self.text_pipeline: Optional[ov_genai.LLMPipeline] = None
        self.embedding_model = None  # OVModelForFeatureExtraction (optional)
        self.embedding_tokenizer = None  # AutoTokenizer (optional)
        self.text_model_path: Optional[str] = None
        self.embedding_model_path: Optional[str] = None
        self.device: Optional[str] = None
        self.stop_generate: bool = False
    
    def ensure_model_downloaded(self, model_id: str, model_path: str) -> str:
        """Download model from Hugging Face if not exists."""
        model_path_obj = Path(model_path)
        
        if model_path_obj.exists():
            # Find the actual model directory (containing .xml files)
            for root, dirs, files in os.walk(model_path_obj):
                if any(f.endswith('.xml') for f in files):
                    return str(Path(root))
            return str(model_path_obj)
        
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")
        
        print(f"Downloading model from Hugging Face: {model_id}")
        print(f"Downloading to: {model_path}")
        print("This may take a few minutes...")
        
        try:
            hf_hub.snapshot_download(model_id, local_dir=model_path)
            print(f"✓ Model downloaded successfully!")
            
            # Find the actual model directory
            for root, dirs, files in os.walk(model_path_obj):
                if any(f.endswith('.xml') for f in files):
                    return str(Path(root))
            return str(model_path_obj)
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    
    def load_text_model(self, model_path: str = None, device: str = None) -> None:
        """Load OpenVINO text generation model (auto-downloads if needed)."""
        if not OPENVINO_GENAI_AVAILABLE:
            raise ImportError("openvino-genai is required. Install with: pip install openvino-genai")
        
        if model_path is None:
            model_path = TEXT_MODEL_PATH
        
        # Auto-download if needed
        actual_model_path = self.ensure_model_downloaded(TEXT_MODEL_ID, model_path)
        
        print(f"Loading OpenVINO text model from: {actual_model_path}")
        
        # Determine device
        if device is None or device.lower() == 'auto':
            device = 'NPU'  # OpenVINO will fallback automatically
            print("→ Auto-selected: NPU (with automatic fallback)")
        else:
            device = device.upper()
            if device not in ['NPU', 'GPU', 'CPU']:
                print(f"⚠ Device '{device}' not recognized, using NPU")
                device = 'NPU'
        
        print(f"✓ Using device: {device}")
        
        try:
            self.text_pipeline = ov_genai.LLMPipeline(actual_model_path, device)
            self.text_model_path = actual_model_path
            if not self.device:
                self.device = device
            self.stop_generate = False
            print("✓ Text model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading text model: {e}")
            raise
    
    def load_embedding_model(self, device: str = None) -> None:
        """Load OpenVINO embedding model (auto-downloads if needed)."""
        if not OPTIMUM_AVAILABLE:
            raise ImportError("optimum-intel is required. Install with: pip install optimum-intel transformers torch")
        
        print(f"Loading OpenVINO embedding model: {EMBEDDING_MODEL_ID}")
        print("This will auto-download from Hugging Face if needed...")
        
        try:
            # from_pretrained handles downloading automatically
            self.embedding_model = OVModelForFeatureExtraction.from_pretrained(
                EMBEDDING_MODEL_ID, 
                export=False
            )
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_ID)
            self.embedding_model_path = EMBEDDING_MODEL_ID
            print("✓ Embedding model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading embedding model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def is_text_loaded(self) -> bool:
        """Check if text model is loaded."""
        return self.text_pipeline is not None
    
    def is_embedding_loaded(self) -> bool:
        """Check if embedding model is loaded."""
        return self.embedding_model is not None
    
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = True
    ) -> Iterator[str]:
        """Generate text from prompt."""
        if not self.is_text_loaded():
            raise RuntimeError("No text model loaded!")
        
        self.stop_generate = False
        
        # Configure generation parameters
        generate_kwargs = {}
        
        if max_length is not None and max_length > 0:
            generate_kwargs['max_length'] = max_length
        
        if temperature > 0:
            generate_kwargs['temperature'] = temperature
            generate_kwargs['top_p'] = top_p
            generate_kwargs['top_k'] = top_k
        
        if stream:
            yield from self._generate_stream(prompt, generate_kwargs)
        else:
            result = self.text_pipeline.generate(prompt, **generate_kwargs)
            yield result if isinstance(result, str) else ''.join(result)
    
    def _generate_stream(self, prompt: str, generate_kwargs: dict) -> Iterator[str]:
        """Generate with streaming."""
        try:
            for token in self.text_pipeline.generate(prompt, **generate_kwargs):
                if self.stop_generate:
                    break
                yield token
        except Exception as e:
            yield f"<error>{str(e)}</error>"
    
    def embed(self, text: str, task_description: str = None) -> list:
        """Generate embedding from text."""
        if not self.is_embedding_loaded():
            raise RuntimeError("No embedding model loaded!")
        
        try:
            # Format text with instruction if provided
            if task_description:
                input_text = f"Instruct: {task_description}\nQuery:{text}"
            else:
                input_text = text
            
            # Tokenize
            max_length = 8192
            batch_dict = self.embedding_tokenizer(
                [input_text],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict.to(self.embedding_model.device)
            
            # Generate embedding
            outputs = self.embedding_model(**batch_dict)
            
            # Pool last token
            def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
                left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings[0].tolist()
        except Exception as e:
            raise RuntimeError(f"Embedding error: {e}")


# Initialize backend
llm_backend = ONNXBackend()


# Define Swagger models
health_model = api.model('Health', {
    'code': fields.Integer(description='Status code'),
    'message': fields.String(description='Message'),
    'text_model_loaded': fields.Boolean(description='Text model loaded'),
    'embedding_model_loaded': fields.Boolean(description='Embedding model loaded'),
    'text_model_path': fields.String(description='Text model path'),
    'embedding_model_path': fields.String(description='Embedding model path'),
    'device': fields.String(description='Device in use')
})

embed_input = api.model('EmbedInput', {
    'text': fields.String(required=True, description='Text to embed', example='Hello world'),
    'task_description': fields.String(required=False, description='Task description for instruction-based embedding', example='Given a web search query, retrieve relevant passages')
})

generate_input = api.model('GenerateInput', {
    'prompt': fields.String(required=True, description='Text prompt', example='Write a haiku about AI'),
    'max_length': fields.Integer(description='Max tokens (None or 0 = no limit)', default=200, required=False),
    'temperature': fields.Float(description='Temperature', default=0.7),
    'top_p': fields.Float(description='Nucleus sampling', default=0.9),
    'top_k': fields.Integer(description='Top-k sampling', default=50)
})

response_model = api.model('Response', {
    'code': fields.Integer(description='Status code'),
    'message': fields.String(description='Message')
})


# Health endpoint
@ns_health.route('')
class Health(Resource):
    @ns_health.marshal_with(health_model)
    def get(self):
        """Check API health and model status"""
        return {
            'code': 0,
            'message': 'success',
            'text_model_loaded': llm_backend.is_text_loaded(),
            'embedding_model_loaded': llm_backend.is_embedding_loaded(),
            'text_model_path': llm_backend.text_model_path,
            'embedding_model_path': llm_backend.embedding_model_path,
            'device': llm_backend.device
        }


# Text generation
@ns_llm.route('/generate')
class Generate(Resource):
    @ns_llm.expect(generate_input)
    def post(self):
        """Generate text (non-streaming)"""
        if not llm_backend.is_text_loaded():
            return {'code': 1, 'message': 'Text model not loaded'}, 400
        
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return {'code': 1, 'message': 'prompt required'}, 400
        
        try:
            max_length = data.get('max_length')
            if max_length == 0:
                max_length = None
            
            result = "".join(llm_backend.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.9),
                top_k=data.get('top_k', 50),
                stream=False
            ))
            
            return {'code': 0, 'message': 'success', 'result': result}
        except Exception as e:
            return {'code': 1, 'message': str(e)}, 500


@ns_llm.route('/chat')
class Chat(Resource):
    @ns_llm.expect(generate_input)
    def post(self):
        """Generate text with streaming (Server-Sent Events)"""
        if not llm_backend.is_text_loaded():
            return {'code': 1, 'message': 'Text model not loaded'}, 400
        
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return {'code': 1, 'message': 'prompt required'}, 400
        
        def generate_stream():
            try:
                max_length = data.get('max_length')
                if max_length == 0:
                    max_length = None
                
                for token in llm_backend.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=data.get('temperature', 0.7),
                    top_p=data.get('top_p', 0.9),
                    top_k=data.get('top_k', 50),
                    stream=True
                ):
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        return Response(generate_stream(), content_type='text/event-stream')


@ns_llm.route('/stop')
class StopGeneration(Resource):
    @ns_llm.marshal_with(response_model)
    def post(self):
        """Stop ongoing generation"""
        llm_backend.stop_generate = True
        return {'code': 0, 'message': 'Generation stopped'}


# Embedding endpoints
@ns_embed.route('/')
class Embed(Resource):
    @ns_embed.expect(embed_input)
    def post(self):
        """Generate embedding from text"""
        if not llm_backend.is_embedding_loaded():
            error_msg = 'Embedding model not loaded'
            if not OPTIMUM_AVAILABLE:
                error_msg += '. Install dependencies: pip install optimum-intel transformers torch'
            else:
                error_msg += '. Check server startup logs for loading errors'
            return {'code': 1, 'message': error_msg}, 400
        
        data = request.get_json()
        text = data.get('text')
        task_description = data.get('task_description')
        
        if not text:
            return {'code': 1, 'message': 'text required'}, 400
        
        try:
            embedding = llm_backend.embed(text, task_description=task_description)
            return {
                'code': 0,
                'message': 'success',
                'embedding': embedding,
                'dimension': len(embedding)
            }
        except Exception as e:
            return {'code': 1, 'message': str(e)}, 500


def main():
    parser = argparse.ArgumentParser(description='OpenVINO GenAI API Server')
    parser.add_argument('--text-model-path', type=str, default=TEXT_MODEL_PATH, help='Path to text model (default: auto-downloads)')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, npu, gpu, cpu')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    
    args = parser.parse_args()
    
    # Auto-load text model (downloads if needed)
    try:
        llm_backend.load_text_model(args.text_model_path, device=args.device)
    except Exception as e:
        print(f"⚠ Failed to load text model: {e}")
        print("   Text generation endpoints will be unavailable")
    
    # Auto-load embedding model (downloads if needed)
    if not OPTIMUM_AVAILABLE:
        print(f"⚠ optimum-intel not available. Embedding model will not load.")
        print("   Install with: pip install optimum-intel transformers torch")
    else:
        try:
            llm_backend.load_embedding_model(device=args.device)
        except Exception as e:
            print(f"⚠ Failed to load embedding model: {e}")
            import traceback
            traceback.print_exc()
            print("   Embedding endpoints will be unavailable")
    
    print(f"\n🚀 OpenVINO GenAI API Server")
    print(f"   Text model: {TEXT_MODEL_ID}")
    print(f"   Embedding model: {EMBEDDING_MODEL_ID}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   Health: http://{args.host}:{args.port}/api/health")
    print(f"   Text model: {'✓ Loaded' if llm_backend.is_text_loaded() else '✗ Not loaded'}")
    print(f"   Embedding model: {'✓ Loaded' if llm_backend.is_embedding_loaded() else '✗ Not loaded'}")
    print()
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
