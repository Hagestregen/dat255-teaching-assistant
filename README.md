## Hosted on:
https://opinions-mainly-informed-regular.trycloudflare.com/

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

```bash
pip install nbstripout
```

In the folder with the .git file (/RegModel) run:

```bash
nbstripout --install
```

## Hugging Face token (HF_TOKEN)

When using models and dowloading them from huggingface, if you are sending an unauthenticated requests to the HF Hub, you can get: 
- lower rate limits
- Downloads can be slower
- Some models may eventually require auth

### Setting HF_TOKEN

Step 1 — Get a token
Go to: https://huggingface.co/settings/tokens
Click “New token”
Choose:
Role: Read
Copy it

install cli (it is installed in the requirements)
pip install huggingface_hub

then
hf auth login

Paste your token.

(Dont need to Add token as git credential? [y/N])

it will be stored in 
~/.cache/huggingface/token


## FAQ

### pip install ipywidgets crashes on windows

**The problem is Windows' default 260-character path limit.**  
The `jupyterlab_widgets` package (part of ipywidgets) installs files with very long paths, which hits this limit → pip fails with "No such file or directory".

1. **Enable Long Paths via PowerShell**  
    Open **PowerShell as Administrator** and run this single command:

    ```powershell
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
    ```

2. Try installing again:

    ```bash
    pip install ipywidgets
    ```

3. **If it dosent work still: Restart your computer**.

#### After Fixing

You can also clean pip's cache if you keep getting errors:

```bash
pip cache purge
pip install ipywidgets --no-cache-dir
```

This should resolve the issue permanently. Let me know if you get any new error after restarting!
