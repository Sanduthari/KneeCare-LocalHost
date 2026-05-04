import os
import json
import uuid
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

VALID_XRAY_LABELS = {
    "knee x-ray",
    "knee_xray",
    "knee-xray",
    "xray",
    "x-ray",
    "knee",
    "knee radiograph",
    "knee radiography",
}

INVALID_XRAY_LABELS = {
    "not x-ray",
    "not_xray",
    "not-xray",
    "non xray",
    "non_xray",
    "non-xray",
    "wrong",
    "invalid",
}

GATE_CONF_THRESHOLD = 0.60

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def display_label(mode: str, raw_label: str) -> str:
    n = normalize_label(raw_label)

    if mode == "mri":
        if n == "abnormal":
            return "Osteoarthritis"
        if n == "normal":
            return "Normal"

    if mode == "xray":
        if n in {"normal", "no oa", "no osteoarthritis", "healthy"}:
            return "Normal"
        if n in {"osteoarthritis", "oa", "abnormal", "arthritis"}:
            return "Osteoarthritis"

    return raw_label


def load_image_for_model(img_path):
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    return img_pil, img_np, img_tensor


def extract_logits_for_prediction(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item

        for item in output:
            if isinstance(item, (tuple, list)):
                for sub in item:
                    if isinstance(sub, torch.Tensor) and sub.ndim == 2:
                        return sub

        for item in output:
            if isinstance(item, torch.Tensor):
                return item.view(item.size(0), -1)

    raise TypeError(f"Could not extract prediction logits from {type(output)}")


def extract_logits_for_cam(output):
    candidates = []

    def collect_tensors(obj):
        if isinstance(obj, torch.Tensor):
            candidates.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                collect_tensors(x)

    collect_tensors(output)

    for t in candidates:
        if t.ndim == 2 and (t.requires_grad or t.grad_fn is not None):
            return t

    for t in candidates:
        if t.requires_grad or t.grad_fn is not None:
            if t.ndim == 2:
                return t
            return t.view(t.size(0), -1)

    for t in candidates:
        if t.ndim == 2:
            return t

    if candidates:
        t = candidates[0]
        if t.ndim == 2:
            return t
        return t.view(t.size(0), -1)

    raise TypeError(f"Could not extract CAM logits from {type(output)}")


class YOLOClsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        logits = extract_logits_for_cam(out)
        return logits


def predict_probs(wrapped_model, img_tensor):
    with torch.no_grad():
        logits = wrapped_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def predict_label(wrapped_model, class_names, img_tensor):
    probs = predict_probs(wrapped_model, img_tensor)
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_conf = float(probs[pred_idx])
    return pred_idx, pred_label, pred_conf, probs


def find_gradcam_target_layer(yolo_model):
    layers = yolo_model.model.model

    print("Searching Grad-CAM target layer...")
    print("YOLO task:", getattr(yolo_model, "task", "unknown"))

    for i, layer in enumerate(layers):
        print(i, layer.__class__.__name__)

    for layer in reversed(layers):
        layer_name = layer.__class__.__name__.lower()

        if "classify" in layer_name:
            continue

        if (
            "conv" in layer_name
            or "c2f" in layer_name
            or "bottleneck" in layer_name
            or "sppf" in layer_name
        ):
            print("Selected Grad-CAM target layer:", layer.__class__.__name__)
            return layer

    print("Fallback Grad-CAM target layer:", layers[-2].__class__.__name__)
    return layers[-2]


def run_yolo_class_prediction(yolo_model, image_path: str):
    results = yolo_model.predict(
        source=image_path,
        imgsz=IMG_SIZE,
        save=False,
        verbose=False
    )

    if not results:
        return {
            "ok": False,
            "error": "No results returned by model"
        }

    r0 = results[0]
    names = getattr(r0, "names", None) or getattr(yolo_model, "names", {})

    if getattr(r0, "probs", None) is not None and r0.probs is not None:
        top_idx = int(r0.probs.top1)
        top_conf = float(r0.probs.top1conf)
        label = names.get(top_idx, str(top_idx)) if isinstance(names, dict) else str(top_idx)

        return {
            "ok": True,
            "label": label,
            "confidence": top_conf,
            "classId": top_idx,
        }

    return {
        "ok": False,
        "error": "Model did not return classification probabilities"
    }


def is_valid_knee_xray(gate_result: dict) -> bool:
    raw_label = gate_result.get("label", "")
    conf = float(gate_result.get("confidence", 0.0))
    label = normalize_label(raw_label)

    valid_norm = {normalize_label(x) for x in VALID_XRAY_LABELS}
    invalid_norm = {normalize_label(x) for x in INVALID_XRAY_LABELS}

    if label in invalid_norm:
        return False

    if label in valid_norm and conf >= GATE_CONF_THRESHOLD:
        return True

    return False


def lime_predict_factory(wrapped_model):
    def lime_predict(images_np):
        batch_tensors = []

        for img in images_np:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            pil_img = Image.fromarray(img).convert("RGB")
            tensor = transform(pil_img)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits = wrapped_model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        return probs

    return lime_predict


def shap_predict_factory(wrapped_model):
    def shap_predict(images_np):
        batch_tensors = []

        for img in images_np:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            pil_img = Image.fromarray(img).convert("RGB")
            tensor = transform(pil_img)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits = wrapped_model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        return probs

    return shap_predict


def save_gradcam(img_path, output_path, wrapped_model, target_layer, class_names):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img_pil, img_np, img_tensor = load_image_for_model(img_path)

    pred_idx, pred_label, pred_conf, probs = predict_label(
        wrapped_model,
        class_names,
        img_tensor
    )

    input_tensor = img_tensor.clone().detach().requires_grad_(True)

    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(img_resized).astype(np.float32) / 255.0

    targets = [ClassifierOutputTarget(pred_idx)]

    for p in wrapped_model.parameters():
        p.requires_grad_(True)

    cam = GradCAM(
        model=wrapped_model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_cam, cmap="jet")
    plt.title(f"Grad-CAM Heatmap\nTarget: {class_names[pred_idx]}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cam_image)
    plt.title(f"Overlay\nPred: {pred_label} ({pred_conf * 100:.2f}%)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    return pred_idx, pred_label, pred_conf, probs


def save_lime(img_path, output_path, wrapped_model, class_names, num_samples=1000):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    img_pil, img_np, img_tensor = load_image_for_model(img_path)

    lime_predict = lime_predict_factory(wrapped_model)

    probs = lime_predict(np.expand_dims(img_np, axis=0))[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_conf = float(probs[pred_idx])

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=lime_predict,
        top_labels=len(class_names),
        hide_color=0,
        num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        label=pred_idx,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    out = mark_boundaries(temp / 255.0, mask)

    plt.figure(figsize=(6, 6))
    plt.imshow(out)
    plt.title(f"LIME\nPred: {pred_label} ({pred_conf * 100:.2f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    return pred_idx, pred_label, pred_conf, probs


def save_shap(img_path, output_path, wrapped_model, class_names, max_evals=300, batch_size=8):
    import shap

    img_pil, img_np, img_tensor = load_image_for_model(img_path)
    x = np.expand_dims(img_np, axis=0)

    shap_predict = shap_predict_factory(wrapped_model)

    probs = shap_predict(x)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    pred_conf = float(probs[pred_idx])

    masker = shap.maskers.Image("blur(32,32)", x[0].shape)
    explainer = shap.Explainer(shap_predict, masker)

    shap_values = explainer(
        x,
        max_evals=max_evals,
        batch_size=batch_size
    )

    plt.figure()
    shap.image_plot(
        [shap_values.values[..., pred_idx]],
        pixel_values=x,
        show=False
    )
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close("all")

    return pred_idx, pred_label, pred_conf, probs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=["xray", "mri"])
    parser.add_argument("--method", required=True, choices=["gradcam", "lime", "shap"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--pred-model", required=True)
    parser.add_argument("--gate-model", default=None)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    try:
        if not os.path.exists(args.image):
            print(json.dumps({
                "ok": False,
                "error": f"Image not found: {args.image}"
            }))
            return

        if not os.path.exists(args.pred_model):
            print(json.dumps({
                "ok": False,
                "error": f"Prediction model not found: {args.pred_model}"
            }))
            return

        if args.mode == "xray" and args.gate_model:
            if not os.path.exists(args.gate_model):
                print(json.dumps({
                    "ok": False,
                    "error": f"Gate model not found: {args.gate_model}"
                }))
                return

            gate_model = YOLO(args.gate_model)
            gate_result = run_yolo_class_prediction(gate_model, args.image)

            if not gate_result.get("ok"):
                print(json.dumps({
                    "ok": False,
                    "error": "Gate model failed",
                    "details": gate_result
                }))
                return

            if not is_valid_knee_xray(gate_result):
                print(json.dumps({
                    "ok": False,
                    "validImage": False,
                    "error": "Wrong image inserted. Please upload a correct knee X-ray image.",
                    "gate": {
                        "label": gate_result.get("label"),
                        "confidence": gate_result.get("confidence"),
                    }
                }))
                return

        base_model = YOLO(args.pred_model)

        if getattr(base_model, "task", None) != "classify":
            print(json.dumps({
                "ok": False,
                "error": "Grad-CAM/LIME/SHAP script currently supports YOLO classification models only.",
                "details": f"Detected model task: {getattr(base_model, 'task', 'unknown')}"
            }))
            return

        net = base_model.model.to(device)
        net.eval()

        if isinstance(base_model.names, dict):
            class_names = {int(k): v for k, v in base_model.names.items()}
        else:
            class_names = {i: v for i, v in enumerate(base_model.names)}

        wrapped_model = YOLOClsWrapper(net).to(device).eval()

        target_layer = find_gradcam_target_layer(base_model)

        os.makedirs(args.output_dir, exist_ok=True)

        filename = f"{args.mode}_{args.method}_{uuid.uuid4().hex[:12]}.png"
        output_path = os.path.join(args.output_dir, filename)

        if args.method == "gradcam":
            pred_idx, pred_label, pred_conf, probs = save_gradcam(
                args.image,
                output_path,
                wrapped_model,
                target_layer,
                class_names
            )

        elif args.method == "lime":
            pred_idx, pred_label, pred_conf, probs = save_lime(
                args.image,
                output_path,
                wrapped_model,
                class_names
            )

        else:
            pred_idx, pred_label, pred_conf, probs = save_shap(
                args.image,
                output_path,
                wrapped_model,
                class_names
            )

        result = {
            "ok": True,
            "mode": args.mode,
            "method": args.method,
            "label": display_label(args.mode, pred_label),
            "rawLabel": pred_label,
            "confidence": float(pred_conf),
            "classId": int(pred_idx),
            "imageUrl": f"/explanations/{filename}",
            "probabilities": {
                display_label(args.mode, class_names[i]): float(probs[i])
                for i in range(len(probs))
            }
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": "XAI generation failed",
            "details": str(e)
        }))


if __name__ == "__main__":
    main()