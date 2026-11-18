import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter

def prepare_digit_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("L")
    except Exception as exc:
        raise ValueError("File is corrupted or not a valid image. Please upload a PNG/JPEG.") from exc

    # 1. Size Check (Agar image bohat choti ya khali hai)
    width, height = image.size
    if width < 20 or height < 20:
        raise ValueError(f"Image is too small ({width}x{height}). Please upload a clear, larger image.")

    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size == 0:
        raise ValueError("Empty image uploaded. Please select a file.")

    # 2. Invert Check (Agar background safed hai to invert karein)
    border = np.concatenate([
        image_array[0, :], image_array[-1, :],
        image_array[:, 0], image_array[:, -1]
    ])
    if np.median(border) > 127:
        image = ImageOps.invert(image)

    image = ImageOps.autocontrast(image)
    image_array = np.asarray(image, dtype=np.float32)

    # 3. Blank Check (Agar image mein koi digit nahi hai)
    max_value = float(image_array.max())
    if max_value < 30: 
        raise ValueError("Blank image detected. Please upload a handwritten digit.")

    # 4. Digit Detection Check
    threshold = max(20.0, max_value * 0.2)
    mask = image_array > threshold
    
    # Agar digit ke pixels bohat kam hain (sirf shor hai)
    if not mask.any() or mask.sum() < 20: 
        raise ValueError("No clear digit found. Please draw a digit (0-9) clearly.")

    rows, cols = np.where(mask)
    left, right = int(cols.min()), int(cols.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    
    # Agar digit image ke barabar hai (poori image bhari hui hai)
    if (right - left) > (width * 0.95) and (bottom - top) > (height * 0.95):
        raise ValueError("Image is too full. Please upload a digit with a clear background.")

    # 5. Crop, Resize & Center
    digit = image.crop((left, top, right, bottom))
    digit = digit.resize((8, 8), Image.Resampling.LANCZOS)
    digit = digit.filter(ImageFilter.GaussianBlur(radius=0.5))

    arr = np.asarray(digit, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Blank image detected after processing.")

    yy, xx = np.indices(arr.shape)
    center_x = float((xx * arr).sum() / total)
    center_y = float((yy * arr).sum() / total)
    shift_x = int(round(3.5 - center_x))
    shift_y = int(round(3.5 - center_y))

    centered = Image.new("L", (8, 8), 0)
    src_left = max(0, -shift_x)
    src_top = max(0, -shift_y)
    src_right = min(8, 8 - shift_x)
    src_bottom = min(8, 8 - shift_y)
    if src_right > src_left and src_bottom > src_top:
        part = digit.crop((src_left, src_top, src_right, src_bottom))
        centered.paste(part, (src_left + shift_x, src_top + shift_y))

    image_array = np.asarray(centered, dtype=np.float32) / 255.0
    if float(image_array.max()) < 0.1:
        raise ValueError("Digit became invisible after resizing. Please upload a clearer image.")

    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float()
    return centered, tensor