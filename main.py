import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-World prediction script")
    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "-c", "--classes",
        type=str,
        nargs="+",
        required=True,
        help="Classes to detect (e.g., -c key watch phone)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolov8x-worldv2.pt",
        help="Path to YOLO-World model (default: yolov8x-worldv2.pt)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and set classes
    model = YOLOWorld(args.model)
    model.set_classes(args.classes)

    # Run prediction
    results = model.predict(source=image_path)

    # Save original image
    original_output = output_dir / f"original{image_path.suffix}"
    shutil.copy(image_path, original_output)

    # Save prediction image
    prediction_output = output_dir / "prediction.jpg"
    results[0].save(str(prediction_output))

    # Save config file
    config = {
        "image": str(image_path),
        "classes": args.classes,
        "model": args.model,
        "timestamp": timestamp,
    }
    config_output = output_dir / "config.json"
    with open(config_output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir}/")
    print(f"  - original: {original_output.name}")
    print(f"  - prediction: {prediction_output.name}")
    print(f"  - config: {config_output.name}")


if __name__ == "__main__":
    main()
