import sys
import os

print("=" * 55)
print("  TOAD × SAM 3 — Environment Verification")
print("=" * 55)

# ── Python ──────────────────────────────────────────────────
print(f"\n[1/5] Python {sys.version.split()[0]}", end="  ")
assert sys.version_info >= (3, 12), "Need Python 3.12+"
print("✓")

# ── PyTorch + device detection ───────────────────────────────
import torch

cuda_ok = torch.cuda.is_available()
mps_ok  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if cuda_ok:
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    note = f"CUDA {torch.version.cuda} — {device_name}"
elif mps_ok:
    device = "mps"
    note = "Apple Silicon MPS (local dev only — run eval/fine-tuning on Colab/HPG)"
else:
    device = "cpu"
    note = "⚠️  CPU only — SAM 3 inference will be very slow"

print(f"[2/5] PyTorch {torch.__version__}  ✓  ({note})")

# ── TensorFlow (SEKO path — must NOT claim the GPU) ──────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
if cuda_ok:
    tf.config.set_visible_devices([], "GPU")
    tf_note = "GPU hidden from TF ✓"
else:
    tf_note = "no GPU to hide"
print(f"[3/5] TensorFlow {tf.__version__}  ✓  ({tf_note})")

# ── OpenCV ───────────────────────────────────────────────────
import cv2
print(f"[4/5] OpenCV {cv2.__version__}  ✓")

# ── SAM 3 ────────────────────────────────────────────────────
print("[5/5] SAM 3 predictor ...", end="  ", flush=True)

try:
    from sam3.model_builder import build_sam3_image_predictor
except ImportError:
    print(
        "\n  ✗  SAM 3 not installed. Run:\n"
        '     pip install "sam3[notebooks,train] @ '
        'git+https://github.com/facebookresearch/sam3.git"'
    )
    sys.exit(1)

CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "sam3.pt")
CFG  = "configs/sam3/sam3_s.yaml"

if not os.path.exists(CKPT):
    print(
        f"\n  ✗  Checkpoint not found at: {CKPT}\n"
        "     Download it with:\n\n"
        "       python -c \"\n"
        "       from huggingface_hub import hf_hub_download\n"
        "       hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt',\n"
        "                       local_dir='./checkpoints')\n"
        "       \"\n"
    )
    sys.exit(1)

predictor = build_sam3_image_predictor(model_cfg=CFG, ckpt_path=CKPT)
print(f"✓  loaded sam3.pt  (device: {device})")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 55)
if device == "mps":
    print("  ✓  Mac dev environment ready.")
    print("  ⚠  For zero-shot eval + fine-tuning, use Colab or")
    print("     HiPerGator (NVIDIA GPU required for full SAM 3).")
elif device == "cuda":
    print("  ✓  All checks passed — ready for Task 5 (prompt gen).")
else:
    print("  ⚠  CPU-only — install complete but inference will be slow.")
print("=" * 55 + "\n")