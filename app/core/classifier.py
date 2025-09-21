from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import open_clip

# Permitir cargar imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo OpenCLIP
clip_model_name = "ViT-B-32"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    clip_model_name, pretrained="openai"
)
clip_model = clip_model.to(device)

# Dataset y DataLoader
train_path = "app/data/test"  # Ajusta a tu ruta real
train_dataset = ImageFolder(train_path, transform=clip_preprocess)

# Filtrar imágenes corruptas
valid_samples = []
for img_path, label in train_dataset.samples:
    try:
        img = Image.open(img_path)
        img.verify()  # Verifica que la imagen se pueda abrir
        valid_samples.append((img_path, label))
    except Exception as e:
        print(f"Imagen corrupta ignorada: {img_path} ({e})")

train_dataset.samples = valid_samples
train_dataset.targets = [label for _, label in valid_samples]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# Función para extraer embeddings
def get_embeddings(dataloader):
    clip_model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = clip_model.encode_image(images)
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_embeddings)
    y = np.concatenate(all_labels)
    return X, y

# Entrenamiento
X_train, y_train = get_embeddings(train_loader)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
print(classification_report(y_train, y_pred, target_names=train_dataset.classes))

