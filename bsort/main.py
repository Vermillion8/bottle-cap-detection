"""CLI entrypoints for the bottle-cap detection pipeline."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import typer
import yaml
from ultralytics import YOLO

import wandb
from bsort.augment import augment_data
from bsort.data import process_data, split_dataset
from bsort.tuner import run_tuner_ui
from bsort.viz import view_labels

app = typer.Typer(help="Bottle Cap Detection Pipeline CLI")


def load_config(config_path: Path) -> Dict:
    """Loads the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_wandb_run(cfg: dict, run_name: str):
    """
    Initializes a WandB run if enabled in the configuration.
    Returns the run handle or None when WandB logging is disabled.
    """
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg is False or wandb_cfg.get("enabled", True) is False:
        return None

    project_name = wandb_cfg.get("project") or cfg["project"]["name"]
    entity = wandb_cfg.get("entity")

    typer.echo(f"Logging metrics to Weights & Biases project '{project_name}'")
    init_kwargs = {
        "project": project_name,
        "config": cfg,
        "name": wandb_cfg.get("run_name") or run_name,
        "reinit": True,
    }
    if entity:
        init_kwargs["entity"] = entity

    try:
        return wandb.init(**init_kwargs)
    except wandb.errors.UsageError as error:
        typer.secho(
            f"Failed to initialize Weights & Biases logging: {error}. "
            "Training will continue without WandB logging.",
            fg=typer.colors.YELLOW,
        )
        return None


def _build_log_payload(row: Dict[str, str]) -> Tuple[Dict[str, float], Optional[int]]:
    """Convert a YOLO CSV row into WandB log data."""
    log_data: Dict[str, float] = {}
    step_value: Optional[int] = None

    epoch_value = row.get("epoch")
    if epoch_value not in (None, ""):
        try:
            step_value = int(float(epoch_value))
        except ValueError:
            step_value = None
        else:
            log_data["epoch"] = step_value

    for key, value in row.items():
        if key == "epoch" or value in (None, ""):
            continue
        try:
            log_data[key] = float(value)
        except ValueError:
            log_data[key] = value

    return log_data, step_value


def log_training_metrics_to_wandb(run_dir: Path, wandb_run) -> None:
    """
    Streams the metrics from Ultralytics' results.csv file into WandB so that
    each training epoch is tracked. Safe to call multiple times.
    """
    if wandb_run is None:
        typer.secho(
            "Skipping WandB metric upload because no run is currently active.",
            fg=typer.colors.YELLOW,
        )
        return

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        typer.secho(
            f"WandB logging skipped: no metrics file found at {results_csv}",
            fg=typer.colors.YELLOW,
        )
        return

    typer.echo(f"Pushing training metrics from {results_csv} to WandB...")
    with results_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row:
                continue
            log_data, step_value = _build_log_payload(row)
            if log_data:
                kwargs = {"step": step_value} if step_value is not None else {}
                wandb_run.log(log_data, **kwargs)

    try:
        wandb_run.save(str(results_csv))
    except OSError as error:
        typer.secho(
            f"Unable to attach results.csv via wandb.save(): {error}. "
            "Falling back to logging it as an artifact.",
            fg=typer.colors.YELLOW,
        )
        try:
            artifact = wandb.Artifact(
                name=f"{wandb_run.name}-metrics",
                type="metrics",
            )
            artifact.add_file(str(results_csv))
            wandb_run.log_artifact(artifact)
        except wandb.errors.CommError as artifact_error:
            typer.secho(
                f"Failed to upload metrics artifact: {artifact_error}",
                fg=typer.colors.RED,
            )


def log_visualizations_to_wandb(run_dir: Path, wandb_run) -> None:
    """
    Sends qualitative visualization images (train batches, labels, predictions,
    confusion matrices, etc.) from the Ultralytics run directory to WandB.
    """
    if wandb_run is None:
        return

    image_specs: Sequence[Tuple[str, str]] = [
        ("train_batch*.jpg", "qualitative/train_batches"),
        ("val_batch*_labels.jpg", "qualitative/val_labels"),
        ("val_batch*_pred.jpg", "qualitative/val_predictions"),
        ("labels.jpg", "qualitative/labels_overlay"),
        ("results.png", "metrics/training_curves"),
        ("confusion_matrix.png", "metrics/confusion_matrix"),
        (
            "confusion_matrix_normalized.png",
            "metrics/confusion_matrix_normalized",
        ),
    ]

    for pattern, log_key in image_specs:
        files = sorted(run_dir.glob(pattern))
        if not files:
            continue

        images: List[wandb.Image] = []
        for path in files[:8]:  # keep uploads lightweight
            try:
                images.append(wandb.Image(str(path), caption=path.name))
            except OSError as error:
                typer.secho(
                    f"Skipping {path} for WandB logging: {error}",
                    fg=typer.colors.YELLOW,
                )

        if images:
            wandb_run.log({log_key: images})


@app.command()
def relabel(
    image_dir: Path = typer.Argument(..., help="Path to source images"),
    label_dir: Path = typer.Argument(..., help="Path to original YOLO labels"),
    output_dir: Path = typer.Argument(..., help="Where to save corrected labels"),
):
    """
    Process images to correct labels based on HSV color detection.
    Distinguishes between light_blue and dark_blue caps.
    """
    typer.echo("Starting data relabeling process...")

    # Call the function from your data.py
    process_data(image_dir, label_dir, output_dir)

    typer.echo("Relabeling complete!")


@app.command()
def tune(
    image_path: Path = typer.Argument(..., help="Path to a single image file to test")
):
    """
    Open a GUI to manually tune HSV color ranges on a specific image.
    """
    if not image_path.exists():
        typer.echo(f"Error: File {image_path} not found.")
        raise typer.Exit(code=1)

    run_tuner_ui(image_path)  # <--- 2. Call the function


@app.command()
def view(
    image_dir: Path = typer.Argument(..., help="Path to source images"),
    label_dir: Path = typer.Argument(..., help="Path to labels (new or old)"),
):
    """
    Visualizes bounding boxes and class names on images.
    """
    if not image_dir.exists() or not label_dir.exists():
        typer.echo("Error: Directories not found.")
        raise typer.Exit(1)

    view_labels(image_dir, label_dir)  # <--- 2. Call the function


@app.command()
def split(
    image_dir: Path = typer.Argument(..., help="Source images"),
    label_dir: Path = typer.Argument(..., help="Source (corrected) labels"),
    output_dir: Path = typer.Argument(..., help="Destination for YOLO dataset"),
    ratio: float = typer.Option(0.8, help="Train split ratio (0.0 - 1.0)"),
):
    """
    Splits images and labels into train/val folders for YOLO.
    """
    split_dataset(image_dir, label_dir, output_dir, ratio)


@app.command()
def augment(
    image_dir: Path = typer.Argument(..., help="Source images"),
    label_dir: Path = typer.Argument(..., help="Source labels"),
    output_dir: Path = typer.Argument(..., help="Folder to save the expanded dataset"),
):
    """
    Multiplies dataset by creating rotated and flipped versions of every image.
    """
    augment_data(image_dir, label_dir, output_dir)


@app.command()
def train(
    config: Path = typer.Option(..., help="Path to settings.yaml"),
):
    """
    Trains the model using parameters from settings.yaml.
    Tracks experiments using Weights & Biases (WandB).
    """
    cfg = load_config(config)
    experiment_name = cfg["project"].get("run_name", "experiment")
    wandb_run = None
    wandb_enabled = cfg.get("wandb", {}).get("enabled", True) is not False

    typer.echo(f"Loading model: {cfg['model']['name']}")
    model = YOLO(cfg["model"]["name"])

    typer.echo("Starting training...")
    model.train(
        data=cfg["data"]["dataset_yaml"],
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["model"]["img_size"],
        batch=cfg["train"]["batch_size"],
        device=cfg["train"]["device"],
        workers=cfg["train"]["workers"],
        mosaic=cfg["train"]["mosaic"],
        project=cfg["project"]["output_dir"],
        name=experiment_name,
        exist_ok=True,
    )

    # Export to ONNX for the inference requirement
    path = model.export(format="onnx", imgsz=cfg["model"]["img_size"])
    typer.echo(f"Model exported to {path}")

    if wandb_enabled:
        run_dir = Path(cfg["project"]["output_dir"]) / experiment_name
        wandb_run = init_wandb_run(cfg, experiment_name)
        try:
            if wandb_run:
                log_training_metrics_to_wandb(run_dir, wandb_run)
                log_visualizations_to_wandb(run_dir, wandb_run)
                wandb_run.log({"artifacts/exported_model": str(path)})
        finally:
            if wandb_run:
                wandb_run.finish()


@app.command()
def infer(
    config: Path = typer.Option(..., help="Path to settings.yaml"),
    image: Path = typer.Option(..., help="Path to image file"),
    model_path: Optional[Path] = typer.Option(
        None, help="Override path to .pt or .onnx file"
    ),
):
    """
    Runs inference on a single image.
    """
    cfg = load_config(config)

    # Default to best.pt if not specified
    if not model_path:
        # Assuming standard YOLO structure
        model_path = Path(cfg["project"]["output_dir"]) / "experiment/weights/best.onnx"

    typer.echo(f"Loading model from: {model_path}")
    model = YOLO(model_path, task="detect")

    results = model.predict(
        source=image,
        imgsz=cfg["model"]["img_size"],
        conf=cfg["inference"]["confidence_threshold"],
        save=True,
    )
    typer.echo(f"Inference done. Saved to {results[0].save_dir}")


if __name__ == "__main__":
    app()
