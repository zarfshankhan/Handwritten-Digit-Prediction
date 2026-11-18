
import torch
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from models import CNNModel, ViTModel

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load original dataset
digits = load_digits()

X = digits.images.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

X = X[:, np.newaxis, :, :]

# Same split as train.py
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Convert test data to tensor
X_test_tensor = torch.tensor(
    X_test, dtype=torch.float32
).to(device)

# Create models
cnn = CNNModel().to(device)

vit = ViTModel(
    patch_size=2,
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    num_classes=10
).to(device)

# Load trained checkpoints
cnn_checkpoint = torch.load(
    "models/cnn_model.pth",
    map_location=device,
    weights_only=True
)

vit_checkpoint = torch.load(
    "models/vit_model.pth",
    map_location=device,
    weights_only=True
)

cnn.load_state_dict(cnn_checkpoint["model_state_dict"])
vit.load_state_dict(vit_checkpoint["model_state_dict"])

cnn.eval()
vit.eval()

# Make predictions
with torch.no_grad():
    cnn_output = cnn(X_test_tensor)
    vit_output = vit(X_test_tensor)

    cnn_predictions = cnn_output.argmax(dim=1).cpu().numpy()
    vit_predictions = vit_output.argmax(dim=1).cpu().numpy()

# Calculate accuracy
cnn_accuracy = accuracy_score(y_test, cnn_predictions)
vit_accuracy = accuracy_score(y_test, vit_predictions)

print("\n========== CNN TEST RESULTS ==========")
print("CNN Accuracy:", round(cnn_accuracy * 100, 2), "%")
print(classification_report(y_test, cnn_predictions, zero_division=0))

print("\n========== VIT TEST RESULTS ==========")
print("ViT Accuracy:", round(vit_accuracy * 100, 2), "%")
print(classification_report(y_test, vit_predictions, zero_division=0))

# Show some incorrect predictions
print("\n========== CNN WRONG PREDICTIONS ==========")

cnn_errors = np.where(cnn_predictions != y_test)[0]

for i in cnn_errors[:10]:
    print(
        "Actual:", y_test[i],
        "| Predicted:", cnn_predictions[i]
    )

print("\n========== VIT WRONG PREDICTIONS ==========")

vit_errors = np.where(vit_predictions != y_test)[0]

for i in vit_errors[:10]:
    print(
        "Actual:", y_test[i],
        "| Predicted:", vit_predictions[i]
    )
