# Doom NWM P0

Doom NWM is a take on the World Model rendering of Doom using a DiT to motivate over a AE representation model, autoregressively tracking over an action-conditioned sequence of frames. 

For more information about it and the development, you can read about it [here](https://seanpatrickmoran.github.io/dit_doom/)

You can call the inference by using the Demos under this repository. demo/raw.sh gives the representation, and you can toggle the trained steering critic from here. or, use demo/flybywire.sh to use the steering model directly.
The release includes two interactive modes:


## Repositories
All the model safetensors can be found at [`signaldetectionweary/doom-nwm-p0`](https://huggingface.co/signaldetectionweary/doom-nwm-p0)

The data needed to start the WM simulation is at [`signaldetectionweary/doom-nwm-p0-bootstraps`](https://huggingface.co/datasets/signaldetectionweary/doom-nwm-p0-bootstraps)

## Quickstart
Use Python 3.10+ with a working PyTorch installation. The inference code additionally requires `numpy`, `opencv-python`, `safetensors`, `huggingface_hub`, and `pyarrow`.

```bash
git clone https://github.com/seanpatrickmoran/doom-nwm-p0.git
cd doom-nwm-p0
```

You don't need an editable package installation. Demo scripts will set `PYTHONPATH` to the repository root.

### Download weights and bootstrap data

The model and bootstrap clips are published in **separate Hugging Face repositories**. The following downloads both into local `release_assets/` directories:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download(
    repo_id="signaldetectionweary/doom-nwm-p0",
    local_dir="release_assets/models",
)
hf_hub_download(
    repo_id="signaldetectionweary/doom-nwm-p0-bootstraps",
    repo_type="dataset",
    filename="bootstraps.parquet",
    local_dir="release_assets/bootstrapped_data",
)
PY
```

### Run raw DiT

```bash
MODEL_ROOT="$PWD/release_assets/models" \
BOOTSTRAP_PATH="$PWD/release_assets/bootstrapped_data/bootstraps.parquet" \
bash demo/raw.sh --ddim_steps 25
```

### Run FlyByWire

```bash
MODEL_ROOT="$PWD/release_assets/models" \
BOOTSTRAP_PATH="$PWD/release_assets/bootstrapped_data/bootstraps.parquet" \
bash demo/flybywire.sh --ddim_steps 25
```

The demos default to a 1280×960 display. `--ddim_steps` controls visible frame generation; FlyByWire's candidate lookahead has its own `--flybywire-ddim-steps` setting (default: 8).

## Controls

| Input | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move |
| Mouse | Turn |
| Left mouse button or `F` | Attack |
| `C` | Toggle steering on/off |
| `Q` or `Esc` | Quit |


## Use Hugging Face directly (optional)
You can also point the model loader at a Hugging Face model ID and download only the bootstrap file into the Hugging Face cache:

```bash
export MODEL_ROOT="signaldetectionweary/doom-nwm-p0"
export BOOTSTRAP_PATH="$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="signaldetectionweary/doom-nwm-p0-bootstrap",
    repo_type="dataset",
    filename="bootstraps.parquet",
))
PY
)"

bash demo/raw.sh --ddim_steps 25
# or
bash demo/flybywire.sh --ddim_steps 25
```

The `MODEL_ROOT` and `BOOTSTRAP_PATH` environment variables override the local defaults in either demo script. There is no need to edit the scripts for this workflow.

If you prefer **Hub-based defaults** inside both scripts, replace their two `MODEL_ROOT` / `BOOTSTRAP_PATH` assignment lines with:

```bash
MODEL_ROOT="${MODEL_ROOT:-signaldetectionweary/doom-nwm-p0}"
BOOTSTRAP_PATH="${BOOTSTRAP_PATH:-$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id="signaldetectionweary/doom-nwm-p0-bootstrap",
    repo_type="dataset",
    filename="bootstraps.parquet",
))
PY
)}"
```

## Project layout

```text
dit/              # VAE, DiT, diffusion, Parquet bootstrap, raw inference
flybywire/        # Steering critic, candidate actions, controller, steered inference
demo/             # raw.sh and flybywire.sh
```

The model is a research prototype; longer rollouts can accumulate visual and dynamical errors. The code and original model contributions are released under the **MIT License**. Bootstrap clips and game assets are distributed separately under their applicable terms.
