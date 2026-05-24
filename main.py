import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# =============================================
# MODEL ARCHITECTURE 
# =============================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, dropout=False):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return torch.relu(self.block(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.e2 = ConvBlock(64,  128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.e5 = ConvBlock(512, 512)
        self.e6 = ConvBlock(512, 512)
        self.e7 = ConvBlock(512, 512)
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        self.d1 = ConvBlock(512,  512, down=False, dropout=True)
        self.d2 = ConvBlock(1024, 512, down=False, dropout=True)
        self.d3 = ConvBlock(1024, 512, down=False, dropout=True)
        self.d4 = ConvBlock(1024, 512, down=False)
        self.d5 = ConvBlock(1024, 256, down=False)
        self.d6 = ConvBlock(512,  128, down=False)
        self.d7 = ConvBlock(256,  64,  down=False)
        self.output = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        b  = self.bottleneck(e7)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        return self.output(torch.cat([d7, e1], dim=1))


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="SAR Image Colorization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = None


def load_model(checkpoint_path: str = "checkpoints/generator_final.pth"):
    """Load generator from checkpoint."""
    global generator
    generator = Generator().to(DEVICE)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(state, dict) and "generator" in state:
            state = state["generator"]
        generator.load_state_dict(state)
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print(f"⚠️ Checkpoint not found at '{checkpoint_path}'.")
    generator.eval()
    

@app.on_event("startup")
def startup_event():
    CHECKPOINT_PATH = "experimentV1/checkpoints/G_epoch_50.pth"
    load_model(CHECKPOINT_PATH)


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING / POSTPROCESSING
# ─────────────────────────────────────────────

IMG_SIZE = 256

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),   # single-channel SAR
])


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [-1, 1] 3-channel tensor → PIL RGB image."""
    img = tensor.squeeze(0).cpu().detach()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Read & preprocess
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    sar_tensor = preprocess(img).mean(dim=0, keepdim=True).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        colorized = generator(sar_tensor)

    # Encode result as PNG
    result_img = tensor_to_pil(colorized)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": generator is not None}


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)