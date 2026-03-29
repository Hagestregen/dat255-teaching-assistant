Yo bitches (skrevet av sivert)

## Installation
### 1. Create and Activate Virtual Environment

```bash
python3 -m venv --prompt dat255_env .venv
source .venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch (GPU)

Install PyTorch with CUDA support **matching your system**.

Check your CUDA version:

```bash
nvidia-smi
```

At the top you should see something like:

```
CUDA Version: 12.6
```

Go to: https://pytorch.org/

Example for CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU is working:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

## Important!!

Jupyter files stores metadata such as execution_count, outputs and others which we dont want to push to github. To filter the metadata out use the library nbstripout,

so before commiting anything jupyter related, install:

```
pip install nbstripout
```

In the folder with the .git file (/RegModel) run:

```
nbstripout --install
```
