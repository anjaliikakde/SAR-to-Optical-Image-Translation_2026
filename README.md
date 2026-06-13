# SAR Image Colorization — Web App

A FastAPI web application that takes a **grayscale SAR image** as input and generates a corresponding **optical RGB image** using a trained Pix2Pix U-Net Generator.

---

## Features

* FastAPI-based inference API
* Web interface for SAR image upload
* Pix2Pix U-Net Generator for SAR-to-optical image translation
* Automatic image preprocessing and postprocessing
* PNG image output
* Support for trained PyTorch checkpoints

---

## Project Structure

```text
sar-colorization/
├── main.py                  ← FastAPI backend + model inference
├── templates/
│   └── index.html           ← Frontend UI
├── checkpoints/
│   └── generator_final.pth  ← Model checkpoint
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

If your checkpoint file has a different name, update the `CHECKPOINT_PATH` variable in `main.py`:

```python
CHECKPOINT_PATH = "checkpoints/your_checkpoint_name.pth"
```

Supported checkpoint formats:

* Raw state dictionary

```python
torch.save(generator.state_dict(), path)
```

* Wrapped dictionary

```python
torch.save({"generator": generator.state_dict()}, path)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python main.py
```

or

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open in Browser

```text
http://localhost:8000
```

---

## Inference Pipeline

1. Upload a SAR image.
2. The image is converted to grayscale and resized to 256×256.
3. Pixel values are normalized to the range `[-1, 1]`.
4. The trained Pix2Pix U-Net Generator performs SAR-to-optical image translation.
5. The generated RGB image is denormalized and returned as a PNG image.
6. The frontend displays both the input and generated output images.

---

## API Endpoints

| Method | Path        | Description                                              |
| ------ | ----------- | -------------------------------------------------------- |
| GET    | `/`         | Serves the web interface                                 |
| POST   | `/colorize` | Accepts an image and returns the generated optical image |
| GET    | `/health`   | Health check endpoint                                    |

---

## Model

### Generator

* Architecture: U-Net Generator (Pix2Pix)
* Input: 1-channel SAR image
* Output: 3-channel RGB optical image
* Resolution: 256 × 256

During inference, only the generator network is required.

---

## Missing Checkpoint

If no checkpoint is found, the application initializes the generator with random weights. The API will still run, but generated outputs will not be meaningful.

---

## Technologies Used

* Python
* FastAPI
* PyTorch
* TorchVision
* Pillow (PIL)
* Uvicorn
* HTML
* CSS

---

## Contributors

* [Anjali Kakde](https://github.com/anjaliikakde)
* [Rushika Gokhe](https://github.com/RushikaGokhe29)
* [Muskan Maran](https://github.com/Muskanmaran18)
