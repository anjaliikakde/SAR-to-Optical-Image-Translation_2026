# SAR Image Colorization — Web App

A FastAPI web application that takes a **grayscale SAR image** as input and returns a **colorized optical image** using your trained Pix2Pix U-Net GAN.

---

## Project Structure

```
sar-colorization/
├── main.py                  ← FastAPI backend + model inference
├── templates/
│   └── index.html           ← Frontend UI
├── checkpoints/
│   └── generator_final.pth  ← ← ← PUT YOUR CHECKPOINT HERE
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Place your checkpoint

Copy your trained generator checkpoint into the `checkpoints/` folder:

```bash
cp /path/to/your/generator_final.pth checkpoints/generator_final.pth
```

If your checkpoint file has a different name, update the `CHECKPOINT_PATH` variable near the bottom of `main.py`:

```python
CHECKPOINT_PATH = "checkpoints/your_checkpoint_name.pth"
```

The app supports two checkpoint formats:
- **Raw state dict**: saved with `torch.save(generator.state_dict(), path)`
- **Wrapped dict**: saved with `torch.save({"generator": generator.state_dict(), ...}, path)`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in browser

```
http://localhost:8000
```

---

## How it works

1. User uploads a SAR image (any format supported by PIL).
2. The backend converts it to grayscale, resizes to 256×256, and normalises to `[-1, 1]`.
3. The U-Net generator produces a 3-channel RGB tensor.
4. The tensor is denormalised and returned as a PNG stream.
5. The frontend displays the original and colorised images side-by-side.

---

## API Endpoints

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/`         | Serves the HTML frontend             |
| POST   | `/colorize` | Accepts `multipart/form-data` image, returns PNG |
| GET    | `/health`   | JSON health check + device info      |

---

## Checkpoint not available?

If the checkpoint file is missing, the model loads with **random weights** and will still run (output will be noise). Replace the checkpoint to get real colorization results.