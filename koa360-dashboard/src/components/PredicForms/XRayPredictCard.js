import React, { useEffect, useMemo, useState } from "react";
import api from "../../api/api";

export default function XRayPredictCard({ open, onClose, patientId, deviceId }) {
  const [mode, setMode] = useState("xray");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [xaiMethod, setXaiMethod] = useState("gradcam");
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiResult, setXaiResult] = useState(null);
  const [xaiError, setXaiError] = useState("");

  useEffect(() => {
    if (!open) return;

    setMode("xray");
    setFile(null);
    setPreviewUrl("");
    setLoading(false);
    setResult(null);
    setError("");

    setXaiMethod("gradcam");
    setXaiLoading(false);
    setXaiResult(null);
    setXaiError("");
  }, [open]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const canSubmit = useMemo(() => !!file && !loading, [file, loading]);

  const canApplyXAI = useMemo(
    () => !!file && !!result && !loading && !xaiLoading,
    [file, result, loading, xaiLoading]
  );

  const buildImageUrl = (path) => {
    if (!path) return "";
    if (path.startsWith("http")) return path;
    return `${api.defaults.baseURL}${path}`;
  };

  const pickFile = (e) => {
    const f = e.target.files?.[0];

    setError("");
    setResult(null);
    setXaiResult(null);
    setXaiError("");

    if (!f) {
      setFile(null);
      setPreviewUrl("");
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const changeMode = (nextMode) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setMode(nextMode);
    setFile(null);
    setPreviewUrl("");
    setResult(null);
    setError("");

    setXaiResult(null);
    setXaiError("");
    setXaiMethod("gradcam");
  };

  const badgeClass = (label) => {
    const t = String(label || "").toLowerCase();

    if (t.includes("osteoarthritis") || t.includes("abnormal") || t.includes("oa")) {
      return "bg-rose-50 border-rose-200 text-rose-800";
    }

    if (t.includes("normal")) {
      return "bg-emerald-50 border-emerald-200 text-emerald-800";
    }

    if (t.includes("invalid") || t.includes("wrong")) {
      return "bg-amber-50 border-amber-200 text-amber-800";
    }

    return "bg-slate-50 border-slate-200 text-slate-800";
  };

  const fmtPct = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return `${(n * 100).toFixed(1)}%`;
  };

  const onSubmit = async () => {
    try {
      setLoading(true);
      setError("");
      setResult(null);
      setXaiResult(null);
      setXaiError("");

      if (!file) {
        setError(`Please select a ${mode === "xray" ? "knee X-ray" : "knee MRI"} image first.`);
        return;
      }

      const fd = new FormData();
      fd.append("image", file);
      fd.append("patientId", patientId || "");
      fd.append("deviceId", deviceId || "");
      fd.append("modality", mode);

      const endpoint = mode === "xray" ? "/api/predict/xray" : "/api/predict/mri";

      const res = await api.post(endpoint, fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (!res?.data) {
        setError("Empty response from server.");
        return;
      }

      if (res.data.ok === false) {
        setError(res.data.error || "Prediction failed");
        return;
      }

      setResult(res.data);
    } catch (e) {
      const msg =
        e?.response?.data?.error ||
        e?.response?.data?.details ||
        e?.message ||
        "Network error";

      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const onApplyXAI = async () => {
    try {
      setXaiLoading(true);
      setXaiError("");
      setXaiResult(null);

      if (!file) {
        setXaiError("Please select an image first.");
        return;
      }

      if (!result) {
        setXaiError("Please run prediction first.");
        return;
      }

      const fd = new FormData();
      fd.append("image", file);
      fd.append("mode", mode);
      fd.append("method", xaiMethod);

      const res = await api.post("/api/explain/image", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (!res?.data) {
        setXaiError("Empty response from XAI server.");
        return;
      }

      if (res.data.ok === false) {
        setXaiError(res.data.error || "XAI generation failed");
        return;
      }

      setXaiResult(res.data);
    } catch (e) {
      const msg =
        e?.response?.data?.error ||
        e?.response?.data?.details ||
        e?.message ||
        "Failed to generate explanation";

      setXaiError(msg);
    } finally {
      setXaiLoading(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-slate-900/50 backdrop-blur-[2px]"
        onClick={!loading && !xaiLoading ? onClose : undefined}
        aria-hidden="true"
      />

      <div
        className="relative w-full max-w-6xl max-h-[90vh] rounded-2xl bg-white border border-slate-200 shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-5 border-b border-slate-200 bg-gradient-to-r from-blue-50 to-white flex items-start justify-between gap-3 shrink-0">
          <div>
            <h3 className="text-lg md:text-xl font-extrabold text-slate-900">
              Medical Image Prediction
            </h3>

            <p className="text-xs text-slate-600 mt-1">
              Upload a knee X-ray or knee MRI image, get prediction, then apply explainable AI.
            </p>

            <div className="mt-2 text-[11px] text-slate-500">
              Patient: <span className="font-extrabold">{patientId || "N/A"}</span>
              {"  "}• Device:{" "}
              <span className="font-extrabold">{deviceId || "N/A"}</span>
            </div>
          </div>

          <button
            onClick={onClose}
            disabled={loading || xaiLoading}
            className="px-3 py-2 rounded-xl border border-slate-200 text-slate-700 text-sm font-extrabold hover:bg-white disabled:opacity-60"
          >
            Close
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="p-5 space-y-5 overflow-y-auto">
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700 text-sm font-bold">
              {error}
            </div>
          )}

          {xaiError && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-amber-700 text-sm font-bold">
              {xaiError}
            </div>
          )}

          {/* Modality Switch */}
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-2 inline-flex gap-2">
            <button
              type="button"
              onClick={() => changeMode("xray")}
              disabled={loading || xaiLoading}
              className={`px-4 py-2 rounded-xl text-sm font-extrabold border transition ${
                mode === "xray"
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
              }`}
            >
              X-ray
            </button>

            <button
              type="button"
              onClick={() => changeMode("mri")}
              disabled={loading || xaiLoading}
              className={`px-4 py-2 rounded-xl text-sm font-extrabold border transition ${
                mode === "mri"
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
              }`}
            >
              MRI
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Upload Section */}
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <div className="flex items-center justify-between">
                <div className="text-sm font-extrabold text-slate-900">
                  Upload {mode === "xray" ? "Knee X-ray" : "Knee MRI"}
                </div>

                <div className="text-xs text-slate-500 font-bold">
                  JPG/PNG/WEBP
                </div>
              </div>

              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={pickFile}
                disabled={loading || xaiLoading}
                className="mt-3 block w-full text-sm"
              />

              <button
                onClick={onSubmit}
                disabled={!canSubmit}
                className="mt-4 w-full px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-extrabold disabled:opacity-60"
              >
                {loading
                  ? "Predicting..."
                  : `Submit & Predict ${mode === "xray" ? "X-ray" : "MRI"}`}
              </button>

              <p className="mt-2 text-xs text-slate-500">
                {mode === "xray"
                  ? "Tip: Upload a clear knee X-ray image."
                  : "Tip: Upload a clear knee MRI image."}
              </p>

              {/* XAI Controls */}
              {result && (
                <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-sm font-extrabold text-slate-900 mb-3">
                    Explainable AI
                  </div>

                  <label className="block text-xs font-bold text-slate-600 mb-2">
                    Select XAI Method
                  </label>

                  <select
                    value={xaiMethod}
                    onChange={(e) => setXaiMethod(e.target.value)}
                    disabled={xaiLoading}
                    className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm"
                  >
                    <option value="gradcam">Grad-CAM</option>
                    <option value="lime">LIME</option>
                    <option value="shap">SHAP</option>
                  </select>

                  <button
                    onClick={onApplyXAI}
                    disabled={!canApplyXAI}
                    className="mt-3 w-full px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-700 text-white text-sm font-extrabold disabled:opacity-60"
                  >
                    {xaiLoading
                      ? "Generating Explanation..."
                      : "Apply Explainable Feature"}
                  </button>

                  <p className="mt-2 text-[11px] text-slate-500">
                    Grad-CAM is usually faster. LIME and SHAP may take longer.
                  </p>
                </div>
              )}
            </div>

            {/* Preview + Prediction */}
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <div className="text-sm font-extrabold text-slate-900">
                Preview
              </div>

              <div className="mt-3 h-52 rounded-2xl border border-slate-200 bg-slate-50 overflow-hidden flex items-center justify-center">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Medical preview"
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <span className="text-sm text-slate-500">
                    No image selected
                  </span>
                )}
              </div>

              <div className="mt-4">
                <div className="text-sm font-extrabold text-slate-900">
                  Prediction Result
                </div>

                {result ? (
                  <div className="mt-2 space-y-2">
                    <div className="text-xs text-slate-500 font-bold uppercase tracking-wide">
                      Modality: {result.modality || mode}
                    </div>

                    <div
                      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-extrabold ${badgeClass(
                        result.label
                      )}`}
                    >
                      <span>{result.label}</span>
                    </div>

                    {result.confidence !== undefined &&
                      result.confidence !== null && (
                        <div className="text-xs text-slate-600">
                          Confidence:{" "}
                          <b className="text-slate-800">
                            {fmtPct(result.confidence)}
                          </b>
                        </div>
                      )}

                    {result.message && (
                      <div className="text-xs text-slate-600">
                        {result.message}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="mt-2 text-sm text-slate-500">
                    Result will appear after prediction.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* XAI Result */}
          {xaiResult && (
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-extrabold text-slate-900">
                    XAI Output
                  </div>

                  <div className="text-xs text-slate-500 mt-1">
                    Method:{" "}
                    <span className="font-extrabold uppercase">
                      {xaiResult.method}
                    </span>
                    {" "}• Label:{" "}
                    <span className="font-extrabold">{xaiResult.label}</span>
                    {" "}• Confidence:{" "}
                    <span className="font-extrabold">
                      {fmtPct(xaiResult.confidence)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Fixed-size XAI Image Viewer */}
              <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 overflow-auto max-h-[560px] p-3 flex items-center justify-center">
                <img
                  src={buildImageUrl(xaiResult.imageUrl)}
                  alt={`${xaiResult.method} explanation`}
                  className="max-w-full max-h-[520px] object-contain rounded-xl"
                />
              </div>

              {xaiResult.probabilities && (
                <div className="mt-4">
                  <div className="text-xs font-extrabold text-slate-700 mb-2">
                    Class Probabilities
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {Object.entries(xaiResult.probabilities).map(([k, v]) => (
                      <div
                        key={k}
                        className="rounded-xl border border-slate-200 px-3 py-2 text-sm flex items-center justify-between"
                      >
                        <span className="font-bold text-slate-700">{k}</span>
                        <span className="text-slate-900 font-extrabold">
                          {fmtPct(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-end gap-2 shrink-0">
          <button
            onClick={onClose}
            disabled={loading || xaiLoading}
            className="px-4 py-2 rounded-xl border border-slate-200 text-slate-700 text-sm font-extrabold hover:bg-white disabled:opacity-60"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}