import sys
import json
import os
from ultralytics import YOLO

# Your trained MRI model classes
# class_to_idx = {"normal": 0, "abnormal": 1}
CLASS_LABEL_MAP = {
    "normal": "Normal",
    "abnormal": "Osteoarthritis"
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def load_model_safe(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MRI model not found: {model_path}")
    return YOLO(model_path)


def run_yolo_prediction(model, image_path: str):
    results = model.predict(
        source=image_path,
        imgsz=224,      # same as your training size
        save=False,
        verbose=False
    )

    if not results:
        return {
            "ok": False,
            "error": "No results returned by MRI model"
        }

    r0 = results[0]
    names = getattr(r0, "names", None) or getattr(model, "names", {})

    # YOLO classification output
    if getattr(r0, "probs", None) is not None and r0.probs is not None:
        top_idx = int(r0.probs.top1)
        top_conf = float(r0.probs.top1conf)

        if isinstance(names, dict):
            label = names.get(top_idx, str(top_idx))
        else:
            label = str(top_idx)

        return {
            "ok": True,
            "type": "classification",
            "label": label,
            "confidence": top_conf,
            "classId": top_idx
        }

    return {
        "ok": False,
        "error": "MRI model did not return classification probabilities"
    }


def format_final_prediction(pred_result: dict):
    raw_label = str(pred_result.get("label", "")).strip()
    normalized = normalize_label(raw_label)
    conf = float(pred_result.get("confidence", 0.0))

    # Map abnormal -> Osteoarthritis, normal -> Normal
    display_label = CLASS_LABEL_MAP.get(normalized, raw_label)

    return {
        "ok": True,
        "validImage": True,
        "modality": "mri",
        "message": "MRI prediction successful",
        "type": pred_result.get("type", "classification"),
        "label": display_label,
        "confidence": conf,
        "classId": pred_result.get("classId", None),
        "rawLabel": raw_label
    }


def main():
    try:
        if len(sys.argv) < 3:
            print(json.dumps({
                "ok": False,
                "error": "Usage: python predict_mri.py <model_path> <image_path>"
            }))
            return

        model_path = sys.argv[1]
        image_path = sys.argv[2]

        if not os.path.exists(model_path):
            print(json.dumps({
                "ok": False,
                "error": f"MRI model not found: {model_path}"
            }))
            return

        if not os.path.exists(image_path):
            print(json.dumps({
                "ok": False,
                "error": f"Image not found: {image_path}"
            }))
            return

        model = load_model_safe(model_path)
        pred_result = run_yolo_prediction(model, image_path)

        if not pred_result.get("ok"):
            print(json.dumps(pred_result))
            return

        print(json.dumps(format_final_prediction(pred_result)))

    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": "MRI prediction failed",
            "details": str(e)
        }))


if __name__ == "__main__":
    main()