"""Helper script to download/convert models for OpenVINO NPU."""

import sys
import subprocess
from pathlib import Path

def check_optimum_installed():
    """Check if optimum-intel is installed."""
    try:
        import optimum
        return True
    except ImportError:
        return False

def install_optimum():
    """Install optimum-intel for OpenVINO model export."""
    print("Installing optimum-intel...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "optimum-intel", "openvino", "nncf"
        ])
        print("✓ Installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False

def export_openvino_model(model_name: str, output_dir: str = None, weight_format: str = "int4"):
    """
    Export a model to OpenVINO IR format for NPU.
    
    Args:
        model_name: Hugging Face model name (e.g., 'microsoft/Phi-3-mini-4k-instruct')
        output_dir: Output directory (default: ./models/<model_name>-openvino-npu)
        weight_format: Weight format ('int4' recommended for NPU)
    """
    if not check_optimum_installed():
        print("optimum-intel not found. Installing...")
        if not install_optimum():
            print("Failed to install optimum-intel. Please install manually:")
            print("  pip install optimum-intel openvino nncf")
            return None
    
    if output_dir is None:
        model_short = model_name.split('/')[-1]
        output_dir = f"./models/{model_short}-openvino-npu"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {model_name} to OpenVINO IR format...")
    print(f"Output directory: {output_path}")
    print("This may take several minutes...")
    
    try:
        # Export model using optimum-cli
        cmd = [
            sys.executable, "-m", "optimum_cli", "export", "openvino",
            "-m", model_name,
            "--weight-format", weight_format,
            "--sym",  # Symmetric quantization
            "--group-size", "-1",  # Channel-wise
            "--ratio", "1.0",
            str(output_path)
        ]
        
        subprocess.check_call(cmd)
        
        print(f"\n✓ Model exported successfully!")
        print(f"  Location: {output_path}")
        
        # Check for .xml and .bin files
        xml_files = list(output_path.rglob("*.xml"))
        bin_files = list(output_path.rglob("*.bin"))
        
        if xml_files and bin_files:
            print(f"  Found {len(xml_files)} XML file(s) and {len(bin_files)} BIN file(s)")
            print(f"  Main model: {xml_files[0]}")
            print(f"\n✓ Model ready for OpenVINO GenAI!")
            print(f"  Use with: python agent_openvino.py \"{output_path}\" \"Your prompt\"")
            return str(output_path)
        else:
            print("⚠ Warning: No .xml/.bin files found")
            return str(output_path)
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Export failed: {e}")
        print("\nTry installing dependencies manually:")
        print("  pip install optimum-intel openvino nncf")
        return None
    except FileNotFoundError:
        print("✗ optimum-cli not found")
        print("Try installing: pip install optimum-intel")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_openvino_model.py <model_name> [output_dir]")
        print("\nExample:")
        print("  python download_openvino_model.py microsoft/Phi-3-mini-4k-instruct")
        print("  python download_openvino_model.py microsoft/Phi-3-mini-4k-instruct ./my_model")
        print("\nNote: This exports the model to OpenVINO IR format (.xml/.bin)")
        print("      which is required for NPU inference.")
        sys.exit(1)
    
    model_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = export_openvino_model(model_name, output_dir)
    
    if result:
        print(f"\n✓ Model ready for NPU at: {result}")
        print("\nNow you can use it with:")
        print(f"  python agent.py \"{result}\" \"Your prompt\"")
        print(f"  python api.py --model \"{result}\"")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

