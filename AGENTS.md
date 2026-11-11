# Project Rules for RETFound

## Python Environment Specification

This project requires all Python execution to run within the dedicated **`retfound`** conda environment.

- **Conda Environment Name**: `retfound`
- **Environment Prefix**: `/workspace/miniconda3/envs/retfound`
- **Python Executable**: `/workspace/miniconda3/envs/retfound/bin/python`
- **Torchrun Executable**: `/workspace/miniconda3/envs/retfound/bin/torchrun`
- **Pip Executable**: `/workspace/miniconda3/envs/retfound/bin/pip`
- **Conda Binary**: `/workspace/miniconda3/bin/conda`
- **Conda Initialization**: `/workspace/miniconda3/etc/profile.d/conda.sh`

---

## Execution Guidelines for Antigravity (agy)

When executing any Python commands, running scripts, training models, launching distributed tasks, or testing within this project:

1. **Always Use the `retfound` Environment**:
   - Explicit binary invocation (preferred for commands and tool calls):
     ```bash
     /workspace/miniconda3/envs/retfound/bin/python <script.py> [args]
     /workspace/miniconda3/envs/retfound/bin/torchrun [args] <script.py> [args]
     ```
   - Conda run invocation:
     ```bash
     conda run -n retfound python <script.py> [args]
     conda run -n retfound torchrun [args] <script.py> [args]
     ```
   - Shell sourcing invocation (for bash scripts and multi-command lines):
     ```bash
     source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate retfound && python <script.py> [args]
     ```
   - PATH prepending (when running shell scripts like `train.sh`):
     ```bash
     PATH="/workspace/miniconda3/envs/retfound/bin:$PATH" bash train.sh
     ```

2. **Prohibited Executables**:
   - DO NOT use the system Python (`/usr/bin/python`, `/usr/bin/python3`).
   - DO NOT use the system torchrun (`/usr/local/bin/torchrun`).
   - DO NOT use the base conda environment (`/workspace/miniconda3/bin/python`).
