# Bottle Cap Detection

`bsort` is an end-to-end machine-learning pipeline that relabels, augments, trains, and deploys a YOLOv8-based detector that classifies conveyor-belt bottle caps (`light_blue`, `dark_blue`, `other`). The repository has a public CLI, configuration-driven training, a reproducible notebook, Docker support, and CI/CD (linting, testing, and container build).

## Highlights
- **Python CLI (`bsort`)** for relabeling, augmentation, dataset splitting, visualization, training, and inference.
- **Configuration-first** workflow driven by `settings.yaml` so datasets, hyperparameters, and devices are centrally managed.
- **Experiment tracking** with Weights & Biases (`wandb.ai`) covering training metrics, artifacts, and latency studies.
- **Reproducible dev environment** defined via `pyproject.toml`, `uv.lock`, Dockerfile, and a detailed Jupyter notebook (`model_development_analysis.ipynb`).
- **CI/CD on GitHub Actions** that enforces pylint, black, isort, pytest, and Docker image builds on every push / PR.

## Repository Layout
```
bottle-cap-detection/
├── bsort/                      # CLI implementation
├── data/, data_augmented/      # Raw and augmented samples
├── dataset_yolo(_aug)/         # YOLO-format splits
├── runs/                       # Training / inference outputs
├── model_development_analysis.ipynb
├── settings.yaml               # Main configuration file
├── pyproject.toml / uv.lock    # Dependency management
├── DockerFile                  # Runtime image
└── .github/workflows/ci.yml    # CI pipeline
```

## Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management (optional but fastest)
- GPU with CUDA 12.1+ (optional, CPU is supported)
- A public [Weights & Biases](https://wandb.ai/) account and API key (`WANDB_API_KEY`)

### Clone & Install
```bash
git clone https://github.com/<your-org>/bottle-cap-detection.git
cd bottle-cap-detection
uv sync        # or: pip install -e .
```

To activate the virtual environment managed by uv:
```bash
uv run python -V
```

## Configuration (`settings.yaml`)
Key sections you can adapt:

| Section      | Keys (examples)                                | Description |
|--------------|------------------------------------------------|-------------|
| `project`    | `name`, `output_dir`                           | WandB project name and local runs folder |
| `data`       | `dataset_yaml`, `source_image_dir`             | Paths to YOLO dataset definition and raw images |
| `model`      | `name`, `img_size`                             | YOLO checkpoint and inference resolution |
| `train`      | `epochs`, `batch_size`, `learning_rate`, etc.  | Training hyperparameters and device |
| `inference`  | `confidence_threshold`, `export_format`        | Serving defaults for CLI inference |

Edit the YAML before running CLI commands to point at your dataset and desired hardware.

## CLI Usage
All functionality is exposed through Typer commands:

```bash
uv run bsort --help
```

| Command | Purpose |
|---------|---------|
| `bsort relabel IMAGE_DIR LABEL_DIR OUTPUT_DIR` | Reassign YOLO labels using HSV heuristics (`bsort/data.py`). |
| `bsort tune IMAGE_PATH` | Launch OpenCV sliders to fine-tune HSV thresholds interactively. |
| `bsort view IMAGE_DIR LABEL_DIR` | Visualize bounding boxes and class IDs for QA. |
| `bsort split IMAGE_DIR LABEL_DIR OUTPUT_DIR --ratio 0.8` | Create YOLO-style `train/val` folders. |
| `bsort augment IMAGE_DIR LABEL_DIR OUTPUT_DIR` | Generate flips/rotations to multiply scarce examples. |
| `bsort train --config settings.yaml` | Train YOLOv8, track to WandB, and export ONNX. |
| `bsort infer --config settings.yaml --image path/to/frame.jpg` | Run single-image prediction with optional `--model-path`. |

All commands include `--help` for parameter details.

## Training Workflow
1. **Authenticate with WandB**
   ```bash
   set WANDB_API_KEY=<your_key>         # Windows PowerShell
   export WANDB_API_KEY=<your_key>      # macOS/Linux
   ```
2. **Prepare data**
   ```bash
   uv run bsort relabel data/images data/labels data/new_labels
   uv run bsort split data/images data/new_labels dataset_yolo --ratio 0.8
   ```
3. **Augment (optional but recommended)**
   ```bash
   uv run bsort augment data/images data/new_labels data_augmented
   ```
4. **Train**
   ```bash
   uv run bsort train --config settings.yaml
   ```
   - Logs, confusion matrices, and weights land in `runs/experiment`.
   - ONNX export path is printed at the end of training.

## Inference & Evaluation
```bash
uv run bsort infer --config settings.yaml --image data/images/raw-250110_dc_s001_b2_15.jpg
```
Predictions (boxes, class IDs, confidence) are saved inside `runs/detect/<timestamp>`. The ONNX artifact enables deployment on edge hardware (e.g., Raspberry Pi 5) with sub-10 ms latency at 320 px inputs as demonstrated in the notebook.

## Notebook
`model_development_analysis.ipynb` documents:
- Data relabeling rationale and HSV thresholds.
- Augmentation strategy and class balance analysis.
- Latency benchmarking of ONNXRuntime.
- Bias/variance findings and ideas for future work.

Run interactively with:
```bash
uv run jupyter notebook model_development_analysis.ipynb
```

## Testing, Linting, and Formatting
Local checks mirror CI:
```bash
uv run black bsort/
uv run isort bsort/
uv run pylint bsort/
uv run pytest
```

## Docker
Build and test the CLI inside a slim runtime image:
```bash
docker build -t bsort:latest .
docker run --rm -v %CD%:/app bsort:latest uv run bsort --help
```
The Dockerfile installs dependencies via uv, validates the CLI during build, and defaults to printing the CLI help text.

## CI/CD
`.github/workflows/ci.yml` executes on every push/PR:
1. Checkout & install uv + Python 3.12.
2. `uv sync` dependencies.
3. `pylint`, `black --check`, `isort --check-only` on `bsort/`.
4. `pytest` for unit tests (pass even when no tests are defined, ensuring framework readiness).
5. Docker image build (`docker build . -t bsort:latest`).

This pipeline guarantees code quality, formatting consistency, import hygiene, test correctness, and container reproducibility.

## Results & Tracking
- Training artifacts and metrics live under `runs/` for offline inspection.
- WandB project (`bottle_cap_sorter`) stores publicly accessible experiment dashboards, including confusion matrices, loss curves, and latency statistics.
- Example validation outputs (confusion matrices, predictions) are stored in `runs/detect/val2` and `runs/detect/predict*`.

## Contributing
1. Fork & clone.
2. Create a feature branch.
3. Run linting/formatting/tests locally.
4. Submit a PR; CI must pass before merge.

## License
MIT © Contributors. See `LICENSE` or `pyproject.toml`.

