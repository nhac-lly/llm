"""
OpenVINO GenAI Agent
Clean CLI tool using OpenVINO GenAI for NPU/GPU/CPU inference.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download

try:
    import openvino_genai as ov_genai
    OPENVINO_GENAI_AVAILABLE = True
except ImportError:
    OPENVINO_GENAI_AVAILABLE = False
    print("⚠ openvino-genai not installed. Install with: pip install openvino-genai")


def check_model_format(model_path: str) -> str:
    """Check if model is OpenVINO IR format."""
    model_path_obj = Path(model_path)
    
    # Check for OpenVINO IR format (.xml + .bin)
    xml_files = list(model_path_obj.rglob("*.xml"))
    bin_files = list(model_path_obj.rglob("*.bin"))
    
    if xml_files and bin_files:
        return "openvino_ir"
    
    return "unknown"


def download_model(repo_id: str, local_dir: str = None, force_download: bool = False):
    """Download an OpenVINO model from Hugging Face Hub."""
    print(f"Downloading OpenVINO model from: {repo_id}")
    
    if local_dir is None:
        model_name = repo_id.split('/')[-1]
        local_dir = f"./models/{model_name}-openvino"
    
    local_path = Path(local_dir)
    
    if local_path.exists() and not force_download:
        print(f"✓ Model already exists at: {local_path}")
        return str(local_path)
    
    os.makedirs(local_path.parent, exist_ok=True)
    
    try:
        print(f"Downloading to: {local_path}")
        print("This may take a few minutes...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )
        
        print(f"✓ Model downloaded successfully!")
        print(f"  Location: {local_path}")
        
        # Find the model directory (should contain .xml and .bin files)
        for root, dirs, files in os.walk(local_path):
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                actual_model_path = Path(root)
                print(f"  Model path: {actual_model_path}")
                return str(actual_model_path)
        
        return str(local_path)
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print(f"     Visit: https://huggingface.co/{repo_id}")
        raise


class GenAIAgent:
    """OpenVINO GenAI agent for NPU/GPU/CPU inference."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the agent.
        
        Args:
            model_path: Path to the OpenVINO model directory (containing .xml/.bin files)
            device: Device preference string:
                   - None or 'auto': Try NPU > GPU > CPU (priority order)
                   - 'npu': Force NPU
                   - 'gpu': Force GPU
                   - 'cpu': Force CPU
        """
        if not OPENVINO_GENAI_AVAILABLE:
            raise ImportError("openvino-genai is required. Install with: pip install openvino-genai")
        
        print(f"Loading OpenVINO model from: {model_path}")
        
        # Check model format
        model_format = check_model_format(model_path)
        if model_format != "openvino_ir":
            raise ValueError(f"Model format '{model_format}' is not OpenVINO IR. Expected .xml/.bin files.")
        
        # Determine device priority
        if device is None or device.lower() == 'auto':
            # Priority: NPU > GPU > CPU
            device = 'NPU'  # OpenVINO will fallback automatically
            print("→ Auto-selected: NPU (with automatic fallback)")
        else:
            device = device.upper()
            if device not in ['NPU', 'GPU', 'CPU']:
                print(f"⚠ Device '{device}' not recognized, using NPU")
                device = 'NPU'
        
        print(f"✓ Using device: {device}")
        print(f"✓ Model format: OpenVINO IR")
        
        # Load model with OpenVINO GenAI
        try:
            self.pipeline = ov_genai.TextPipeline(model_path, device=device)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        self.model_path = model_path
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = True
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate (None or 0 = no limit)
            temperature: Sampling temperature (0=greedy)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stream: Whether to stream output
        """
        # Configure generation parameters
        generate_kwargs = {}
        
        if max_length is not None and max_length > 0:
            generate_kwargs['max_new_tokens'] = max_length
        
        if temperature > 0:
            generate_kwargs['temperature'] = temperature
            generate_kwargs['top_p'] = top_p
            generate_kwargs['top_k'] = top_k
        
        if stream:
            return self._generate_stream(prompt, generate_kwargs)
        else:
            return self._generate_batch(prompt, generate_kwargs)
    
    def _generate_stream(self, prompt: str, generate_kwargs: dict) -> str:
        """Generate with streaming."""
        print(prompt, end='', flush=True)
        generated_text = prompt
        
        try:
            for token in self.pipeline.generate(prompt, **generate_kwargs):
                print(token, end='', flush=True)
                generated_text += token
            print()
        except Exception as e:
            print(f"\n✗ Generation error: {e}")
            raise
        
        return generated_text
    
    def _generate_batch(self, prompt: str, generate_kwargs: dict) -> str:
        """Generate in batch mode."""
        try:
            result = self.pipeline.generate(prompt, **generate_kwargs)
            return result if isinstance(result, str) else ''.join(result)
        except Exception as e:
            print(f"✗ Generation error: {e}")
            raise


def interactive_mode(agent: GenAIAgent):
    """Interactive chat mode."""
    print("\n=== Interactive Mode ===")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                break
            
            if not prompt:
                continue
            
            print("Agent: ", end='', flush=True)
            agent.generate(prompt, stream=True)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenVINO GenAI Agent')
    parser.add_argument('model_path', nargs='?', help='Path to OpenVINO model directory')
    parser.add_argument('prompt', nargs='?', help='Text prompt to generate')
    parser.add_argument('--download', type=str, help='Download model from Hugging Face (repo_id)')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, npu, gpu, cpu')
    parser.add_argument('--max-length', type=int, default=None, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming output')
    
    args = parser.parse_args()
    
    # Download model if requested
    if args.download:
        model_path = download_model(args.download)
        if not args.model_path:
            args.model_path = model_path
    
    if not args.model_path:
        parser.print_help()
        print("\n⚠ Error: model_path is required")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = GenAIAgent(args.model_path, device=args.device)
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(agent)
        return
    
    # Single generation
    if args.prompt:
        agent.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=not args.no_stream
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
