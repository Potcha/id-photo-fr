from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO
from PIL import Image
import numpy as np, json, math

app = FastAPI(title="ID Photo FR – Backend API", version="1.1.0")

# CORS: ouvert en dev (restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DPI_DEFAULT = 300
PX_PER_MM_DEFAULT = DPI_DEFAULT / 25.4
FACE_MIN_MM, FACE_MAX_MM, FACE_TARGET_MM = 32.0, 36.0, 34.0
EYES_TOL_DEG = 2.0

class Point(BaseModel):
    x: float
    y: float

class ValidationRequest(BaseModel):
    chin: Optional[Point] = None
    crown: Optional[Point] = None
    eyeL: Optional[Point] = None
    eyeR: Optional[Point] = None
    rotation_deg: float = 0.0
    scale: float = 1.0
    dpi: int = DPI_DEFAULT

class ValidationResult(BaseModel):
    face_mm: Optional[float]
    face_px_canvas: Optional[float]
    face_ok: bool
    eyes_angle_deg: Optional[float]
    eyes_ok: Optional[bool]
    compliant: bool
    details: dict

def _rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """sRGB -> CIE Lab (D65)"""
    srgb = img_rgb.astype(np.float32) / 255.0
    below = srgb <= 0.04045
    srgb[below] = srgb[below] / 12.92
    srgb[~below] = ((srgb[~below] + 0.055) / 1.055) ** 2.4
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)
    XYZ = np.tensordot(srgb, M.T, axes=1)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = XYZ[...,0] / Xn
    Y = XYZ[...,1] / Yn
    Z = XYZ[...,2] / Zn
    eps = 216/24389
    kappa = 24389/27
    def f(t):
        t = np.asarray(t)
        a = np.cbrt(np.maximum(t, eps))
        b = (kappa*t + 16.0) / 116.0
        return np.where(t > eps, a, b)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L,a,b], axis=-1).astype(np.float32)

def _mask_from_samples(img_rgb: np.ndarray, samples_xy: List[Point], tol: float, soft: float) -> np.ndarray:
    """Retourne alpha (255 sujet, 0 fond) via min distance ΔE(Lab) aux échantillons."""
    H, W, _ = img_rgb.shape
    lab = _rgb_to_lab(img_rgb)
    pts = []
    for p in samples_xy:
        x = int(round(np.clip(p.x, 0, W-1)))
        y = int(round(np.clip(p.y, 0, H-1)))
        pts.append(lab[y, x, :])
    samples_lab = np.stack(pts, axis=0)
    lab_px = lab.reshape(-1,3)
    de = np.sqrt(np.sum((lab_px[:,None,:] - samples_lab[None,:,:])**2, axis=-1))
    de_min = de.min(axis=1).reshape(H,W)
    t0 = max(0.0, tol - soft)
    t1 = tol + soft + 1e-6
    a = np.zeros_like(de_min, dtype=np.float32)
    a[de_min >= t1] = 255.0
    mid = (de_min > t0) & (de_min < t1)
    a[mid] = 255.0 * ((de_min[mid] - t0) / (t1 - t0))
    return a.astype(np.uint8)

@app.get("/health")
def health():
    return {"ok": True, "version": app.version}

@app.post("/validate", response_model=ValidationResult)
def validate(req: ValidationRequest):
    px_per_mm = (req.dpi or DPI_DEFAULT) / 25.4
    face_px_canvas = None
    face_mm = None
    face_ok = False
    details = {}
    if req.chin and req.crown:
        dx = req.crown.x - req.chin.x
        dy = req.crown.y - req.chin.y
        d_img = (dx*dx + dy*dy) ** 0.5
        face_px_canvas = d_img * (req.scale or 1.0)
        face_mm = face_px_canvas / px_per_mm if px_per_mm > 0 else None
        if face_mm is not None:
            face_ok = (FACE_MIN_MM <= face_mm <= FACE_MAX_MM)
        details.update({"face_px_canvas": face_px_canvas, "face_mm": face_mm, "target_mm": FACE_TARGET_MM})
    eyes_angle_deg = None
    eyes_ok = None
    if req.eyeL and req.eyeR:
        ang_raw = math.degrees(math.atan2(req.eyeR.y - req.eyeL.y, req.eyeR.x - req.eyeL.x))
        eyes_angle_deg = ang_raw + (req.rotation_deg or 0.0)
        eyes_ok = abs(eyes_angle_deg) <= EYES_TOL_DEG
        details.update({"eyes_angle_raw_deg": ang_raw, "rotation_deg": req.rotation_deg})
    compliant = bool(face_ok and (eyes_ok is None or eyes_ok))
    return ValidationResult(
        face_mm=face_mm,
        face_px_canvas=face_px_canvas,
        face_ok=face_ok,
        eyes_angle_deg=eyes_angle_deg,
        eyes_ok=eyes_ok,
        compliant=compliant,
        details=details
    )

@app.post("/mask/pipette")
async def mask_pipette(file: UploadFile = File(...), payload: str = Form(...)):
    try:
        data = json.loads(payload)
    except Exception:
        return JSONResponse({"error": "Invalid payload JSON"}, status_code=400)
    samples = [Point(**p) for p in data.get("samples", [])]
    if not samples:
        return JSONResponse({"error": "Provide at least one sample"}, status_code=400)
    tol = float(data.get("tolerance", 18))
    soft = float(data.get("softness", 4))
    im = Image.open(file.file).convert("RGB")
    alpha = _mask_from_samples(np.array(im), samples, tol, soft)
    H, W = alpha.shape
    rgba = np.zeros((H,W,4), dtype=np.uint8)
    rgba[...,0:3] = 255
    rgba[...,3] = alpha
    out = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
