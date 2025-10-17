import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pth"  # caminho do seu modelo .pth
BASE_DIR = "crack_segmentation_dataset/test"  # ajuste esse caminho
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKS_DIR = os.path.join(BASE_DIR, "masks")
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# DATASET SIMPLES
# -----------------------------
class CrackSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask


# -----------------------------
# CARREGAR MODELO E DADOS
# -----------------------------
test_dataset = CrackSegmentationDataset(IMAGES_DIR, MASKS_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# MÉTRICAS
# -----------------------------
f1_metric = smp.utils.metrics.Fscore()
iou_metric = smp.utils.metrics.IoU()
precision_metric = smp.utils.metrics.Precision()
recall_metric = smp.utils.metrics.Recall()

f1s, ious, precisions, recalls = [], [], [], []

# -----------------------------
# AVALIAR E GERAR IMAGENS
# -----------------------------
for idx, (image, mask) in enumerate(tqdm(test_loader)):
    image, mask = image.to(DEVICE), mask.to(DEVICE)
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    f1s.append(f1_metric(pred, mask).item())
    ious.append(iou_metric(pred, mask).item())
    precisions.append(precision_metric(pred, mask).item())
    recalls.append(recall_metric(pred, mask).item())

    if idx < 5:  # salva as 5 primeiras imagens
        img_np = np.transpose(image.cpu().squeeze().numpy(), (1, 2, 0))
        mask_np = mask.cpu().squeeze().numpy()
        pred_np = pred.cpu().squeeze().numpy()

        overlay = np.zeros_like(img_np)
        overlay[..., 1] = mask_np * 255  # verde: real
        overlay[..., 0] = pred_np * 255  # vermelho: predição

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Imagem Original")
        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Máscara Real")
        axs[2].imshow(pred_np, cmap="gray")
        axs[2].set_title("Predição")
        axs[3].imshow(img_np)
        axs[3].imshow(overlay, alpha=0.4)
        axs[3].set_title("Sobreposição (verde=real, vermelho=pred.)")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/resultado_{idx}.png")
        plt.close(fig)

# -----------------------------
# RESULTADOS FINAIS
# -----------------------------
print(f"\n--- MÉTRICAS MÉDIAS ---")
print(f"F1: {np.mean(f1s):.4f}")
print(f"IoU: {np.mean(ious):.4f}")
print(f"Precisão: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")

# Gráfico resumido das métricas
plt.boxplot([f1s, ious, precisions, recalls],
            labels=["F1", "IoU", "Precisão", "Recall"])
plt.title("Distribuição das Métricas de Segmentação")
plt.ylabel("Valor")
plt.grid(True)
plt.savefig(f"{RESULTS_DIR}/metricas_boxplot.png")
plt.close()
