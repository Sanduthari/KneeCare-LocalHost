const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const router = express.Router();

const UPLOAD_DIR = path.join(__dirname, "..", "uploads", "xai");
const OUTPUT_DIR = path.join(__dirname, "..", "public", "explanations");

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, UPLOAD_DIR),
  filename: (_, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `xai_${Date.now()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 12 * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.mimetype)) {
      return cb(new Error("Invalid image type. Only JPG, PNG, and WEBP are allowed."), false);
    }
    cb(null, true);
  },
});

const PYTHON_EXE =
  process.platform === "win32"
    ? path.join(__dirname, "..", "pyenv", "Scripts", "python.exe")
    : "python3";

const PY_SCRIPT = path.join(__dirname, "..", "python", "explain_image.py");

// Adjust these paths to your actual files
const XRAY_GATE_MODEL = path.join(__dirname, "..", "models", "gate.pt");
const XRAY_PRED_MODEL = path.join(__dirname, "..", "models", "best.pt");
const MRI_PRED_MODEL = path.join(__dirname, "..", "models", "Yolo(MRI).pt");

router.post("/api/explain/image", upload.single("image"), (req, res) => {
  try {
    if (!req.file?.path) {
      return res.status(400).json({
        ok: false,
        error: "Image file is required. Use form-data field name: image",
      });
    }

    const mode = String(req.body.mode || "").trim().toLowerCase();
    const method = String(req.body.method || "").trim().toLowerCase();

    if (!["xray", "mri"].includes(mode)) {
      try { fs.unlinkSync(req.file.path); } catch (_) {}
      return res.status(400).json({
        ok: false,
        error: "Invalid mode. Use xray or mri.",
      });
    }

    if (!["gradcam", "lime", "shap"].includes(method)) {
      try { fs.unlinkSync(req.file.path); } catch (_) {}
      return res.status(400).json({
        ok: false,
        error: "Invalid method. Use gradcam, lime, or shap.",
      });
    }

    const imgPath = path.resolve(req.file.path);
    const scriptAbs = path.resolve(PY_SCRIPT);
    const outputAbs = path.resolve(OUTPUT_DIR);

    const args = [
      scriptAbs,
      "--mode", mode,
      "--method", method,
      "--image", imgPath,
      "--output-dir", outputAbs,
    ];

    if (mode === "xray") {
      args.push("--pred-model", path.resolve(XRAY_PRED_MODEL));
      args.push("--gate-model", path.resolve(XRAY_GATE_MODEL));
    } else {
      args.push("--pred-model", path.resolve(MRI_PRED_MODEL));
    }

    const py = spawn(PYTHON_EXE, args, {
      cwd: path.join(__dirname, ".."),
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (d) => (stdout += d.toString()));
    py.stderr.on("data", (d) => (stderr += d.toString()));

    py.on("error", (err) => {
      try { fs.unlinkSync(imgPath); } catch (_) {}
      return res.status(500).json({
        ok: false,
        error: "Failed to start XAI python process",
        details: err.message,
      });
    });

    py.on("close", (code) => {
      try { fs.unlinkSync(imgPath); } catch (_) {}

      if (code !== 0) {
        return res.status(500).json({
          ok: false,
          error: "XAI python process failed",
          exitCode: code,
          stderr: stderr || null,
          stdout: stdout || null,
        });
      }

      try {
        const raw = (stdout || "").trim();
        const lines = raw.split("\n").map((x) => x.trim()).filter(Boolean);
        const jsonLine = lines[lines.length - 1];
        const data = JSON.parse(jsonLine);

        if (data.ok === false) {
          return res.status(500).json(data);
        }

        return res.status(200).json(data);
      } catch (e) {
        return res.status(500).json({
          ok: false,
          error: "Invalid JSON output from XAI python script",
          details: stderr || stdout || e.message,
        });
      }
    });
  } catch (e) {
    return res.status(500).json({
      ok: false,
      error: e.message,
    });
  }
});

module.exports = router;