import argparse
import os
import subprocess

from transformers import AutoConfig

# The 8bit, 4bit, ... models will not work (see README)
# Predefined model aliases
MODELS_ALIASES = {
    # "qwq-32b-8bit": "mlx-community/QwQ-32B-Preview-8bit",
    # "qwen-32b-8bit": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit",
    # "llama-70b-8bit": "mlx-community/Llama-3.3-70B-Instruct-8bit",
    # "deepseek-70b-8bit": "mlx-community/DeepSeek-R1-Distill-Llama-70B-8bit",
    "mlx-deepseek-32b": "mlx-community/DeepSeek-R1-Distill-Qwen-32B",
    "llama-3b": "mlx-community/Llama-3.2-3B-Instruct",
}

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Llama model download, conversion, and inference script.")
    parser.add_argument("--model", type=str, required=True,
                        help="Select a predefined model alias (e.g., 'deepseek-70b', 'qwen-32b') or specify a full Hugging Face model name.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading the model if it already exists.")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip GGUF conversion if it already exists.")
    parser.add_argument("--run-model", action="store_true", help="Run the model after conversion for testing.")
    parser.add_argument("--verbose", action="store_true", help="Display verbose output during model inference.")
    return parser.parse_args()

args = parse_args()

# Resolve model alias or use full model name
MODEL_NAME = MODELS_ALIASES.get(args.model, args.model)

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DOWNLOAD_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_SHORTNAME = MODEL_NAME.split("/")[-1]
SAVE_DIR = os.path.join(DOWNLOAD_DIR, MODEL_SHORTNAME)
LLAMA_CPP_DIR = os.path.join(SCRIPT_DIR, "llama.cpp")

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure necessary directories exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Fetch model metadata
def fetch_model_metadata():
    print(f"Fetching model metadata for {MODEL_NAME}...")

    try:
        config = AutoConfig.from_pretrained(MODEL_NAME)
        quantization = getattr(config, "quantization_config", {}).get("quant_method", "f16")
        model_type = getattr(config, "model_type", "unknown")

        print(f"Detected model type: {model_type}, quantization: {quantization}")
        return model_type, quantization
    except Exception as e:
        print(f"Error fetching model metadata: {e}")
        return None, None

MODEL_TYPE, QUANTIZATION_TYPE = fetch_model_metadata()
if not MODEL_TYPE or not QUANTIZATION_TYPE:
    raise ValueError("Failed to retrieve model metadata. Ensure the model exists on Hugging Face and has a valid config.")

GGUF_PATH = os.path.join(SAVE_DIR, f"{MODEL_SHORTNAME}-{QUANTIZATION_TYPE.upper()}.gguf")

def download_model():
    if not args.skip_download and not os.path.exists(SAVE_DIR):
        print(f"Downloading model {MODEL_NAME} to {SAVE_DIR}...")

        os.makedirs(SAVE_DIR, exist_ok=True)

        subprocess.run([
            "huggingface-cli", "download", MODEL_NAME, "--local-dir", SAVE_DIR, "--resume-download"
        ], check=True)

        print(f"Model downloaded successfully to {SAVE_DIR}.")
    else:
        print(f"Model already exists in {SAVE_DIR}, skipping download.")

def check_model_files():
    pytorch_model_files = [f for f in os.listdir(SAVE_DIR) if f.endswith((".bin", ".safetensors"))]
    if len(pytorch_model_files) > 1:
        print("Multiple model checkpoint files detected. Merging weights may be required.")

def update_llama_cpp():
    if not os.path.exists(LLAMA_CPP_DIR):
        print("Cloning llama.cpp repository...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", LLAMA_CPP_DIR], check=True)
    else:
        print("Updating llama.cpp repository...")
        subprocess.run(["git", "-C", LLAMA_CPP_DIR, "pull"], check=True)
    print("Checking out the latest stable release of llama.cpp...")
    subprocess.run(["git", "-C", LLAMA_CPP_DIR, "checkout", "master"], check=True)

def convert_model():
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"Error: Could not find {convert_script}. Please update llama.cpp repository.")

    if not args.skip_conversion and not os.path.exists(GGUF_PATH):
        print(f"Converting model to GGUF format with quantization {QUANTIZATION_TYPE}...")

        subprocess.run(["python3", convert_script, SAVE_DIR, "--outtype", QUANTIZATION_TYPE], check=True)

        print(f"Conversion completed. GGUF file saved at: {GGUF_PATH}")
    else:
        print(f"GGUF model already exists at {GGUF_PATH}, skipping conversion.")


def run_model():
    if not args.run_model:
        print("Skipping model test run as --run-model is not specified.")
        return

    print("Running GGUF model with Metal GPU acceleration...")

    if os.path.exists(GGUF_PATH):
        llama_bin = os.path.join(LLAMA_CPP_DIR, "build/bin/llama-cli")

        n_gpu_layers_mapping = {
            "72B": "40",
            "32B": "60",
            "14B": "70",
            "12B": "70",
            "7B":  "80",
            "3B":  "90",
            "1B": "100",
        }

        # Extract model size from MODEL_NAME
        model_size = next((size for size in n_gpu_layers_mapping if size in MODEL_NAME), None)
        n_gpu_layers = n_gpu_layers_mapping.get(model_size, "30")

        print(f"Using {n_gpu_layers} GPU layers for model: {MODEL_NAME}")

        cmd = [
            llama_bin,
            "-m", GGUF_PATH,
            "--n-gpu-layers", n_gpu_layers,
            "--ctx-size", "8192",
            "-p", "Write a 1000-word story."
        ]

        subprocess.run(cmd, check=True)
    else:
        raise FileNotFoundError(f"Error: GGUF model not found at {GGUF_PATH}. Ensure conversion was successful.")

if __name__ == "__main__":
    download_model()
    check_model_files()
    update_llama_cpp()
    convert_model()
    run_model()