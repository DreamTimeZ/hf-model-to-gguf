# Convert Hf model to gguf

**IMPORTANT: The project is completely optimized for macOS. You need to adjust it for other operating systems.**

## Installation

### Install Python

- We use version: `3.13.2`

#### Windows

```text
winget install --id=Python.Python.3.13 -e --interactive
```

#### MacOS

```text
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```text
brew install python@3.13
```

### Install cmake

#### Windows

```text
winget install --id=Kitware.CMake  -e
```

#### MacOS

```text
brew install cmake
```

### Clone the repository

```text
git clone git@github.com:DreamTimeZ/hf-model-to-gguf.git
```

### Create Virtual Python Environment (venv)

#### PyCharm

##### Windows: Get Python Binary Path

```powershell
(Get-Command python).Source
```

##### macOS / Linux: Get Python Binary Path

```bash
which python || command -v python3
```

- `shift + shift` -> Search for: `VirtualEnv` -> Create VirtualEnv -> Click Python Interpreter Combo box -> 
  Show All -> + -> Add Local Interpreter... -> Base python -> Paste the path from before

#### Commandline

```bash
python -m venv .venv
```

### Install dependencies

```text
python -m pip install --upgrade pip
```

```text
pip install -r requirements.txt
```

#### Pip Warning

`WARNING: There was an error checking the latest version of pip.`

##### Solution

- [Stackoverflow](https://stackoverflow.com/questions/72439001/there-was-an-error-checking-the-latest-version-of-pip)

###### Windows

```powershell
Remove-Item -Recurse $env:LOCALAPPDATA\pip\cache\selfcheck\
```

###### macOS

```bash
rm -r ~/Library/Caches/pip/selfcheck/
```

###### Linux

```bash
rm -r ~/.cache/pip/selfcheck/
```

### Build llama.cpp

```text
cd llama.cpp
mkdir -p build
cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release
```

## Configuration

### Recommended `n_gpu_layers` Settings for M4 Max (128GB RAM)

| Model Size  | `n_gpu_layers` Recommendation |
|-------------|-------------------------------|
| **3B - 7B** | `--n-gpu-layers 100`          |
| **14B**     | `--n-gpu-layers 80`           |
| **32B**     | `--n-gpu-layers 60`           |
| **70B+**    | `--n-gpu-layers 40`           |

## Errors

### Can not map tensor 'model.embed_tokens.biases'

- https://github.com/ggml-org/llama.cpp/discussions/8521
  - If `model.embed_tokens.weight` is pre-quantized the convertion will not work (not supported yet by the llama.cpp converter).
- Meaning 8bit, 4bit, ... models cannot be converted (only the default ones work). But `mlx` works.