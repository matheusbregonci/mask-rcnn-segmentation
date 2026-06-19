"""Dataset e transforms para Mask R-CNN no formato de anotação YOLO-seg.

Cada arquivo de label .txt tem uma instância por linha:

    class_id x1 y1 x2 y2 ... xn yn   (coordenadas de polígono normalizadas 0-1)

Os polígonos são rasterizados em máscaras binárias e convertidos para o
formato de target esperado pelo Mask R-CNN do torchvision.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.ops import masks_to_boxes
from torchvision.transforms import functional as F

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class YoloSegDataset(torch.utils.data.Dataset):
    """Lê pares imagem/label (YOLO-seg) e produz targets do torchvision.

    target = {
        boxes  FloatTensor[N, 4]  (x1, y1, x2, y2) em pixels,
        labels Int64Tensor[N]     (classe + 1; 0 é reservado p/ background),
        masks  UInt8Tensor[N, H, W],
        image_id, area, iscrowd
    }
    """

    def __init__(self, images_dir, labels_dir=None, transforms=None):
        self.images_dir = Path(images_dir)
        if labels_dir is None:
            labels_dir = self.images_dir.parent / "labels"
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {self.images_dir}")

        self.images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.images:
            raise FileNotFoundError(f"Nenhuma imagem em {self.images_dir}")

    def __len__(self):
        return len(self.images)

    def _label_path(self, img_path):
        return self.labels_dir / f"{img_path.stem}.txt"

    def _parse_label(self, label_path, w, h):
        boxes, labels, masks = [], [], []
        if not label_path.exists():
            return boxes, labels, masks

        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 7:  # classe + ao menos 3 pontos (x, y)
                continue

            cls = int(float(parts[0]))
            coords = np.array(parts[1:], dtype=np.float32)
            coords = coords[: (len(coords) // 2) * 2].reshape(-1, 2)

            poly = coords.copy()
            poly[:, 0] *= w
            poly[:, 1] *= h
            poly = poly.astype(np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cls + 1)  # +1: 0 = background no torchvision
            masks.append(mask)

        return boxes, labels, masks

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels, masks = self._parse_label(self._label_path(img_path), w, h)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels_t),), dtype=torch.int64),
        }

        img_t = F.to_tensor(img)  # [C, H, W] float 0-1
        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)
        return img_t, target


def collate_fn(batch):
    """Mantém imagens de tamanhos diferentes como listas (exigência do modelo)."""
    return tuple(zip(*batch))


# --------------------------------------------------------------------------- #
# Transforms (operam sobre tensor de imagem + target em conjunto)
# --------------------------------------------------------------------------- #
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1).item() < self.p:
            img = F.hflip(img)
            w = img.shape[-1]
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if target["masks"].numel():
                target["masks"] = target["masks"].flip(-1)
        return img, target


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1).item() < self.p:
            img = F.vflip(img)
            h = img.shape[-2]
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
            if target["masks"].numel():
                target["masks"] = target["masks"].flip(-2)
        return img, target


class RandomRotation90:
    """Rotaciona 0/90/180/270°. Ideal para objetos sem orientação canônica
    (ex.: grânulos): a máscara gira sem interpolação e as boxes são
    recalculadas a partir da máscara rotacionada."""

    def __call__(self, img, target):
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return img, target
        img = torch.rot90(img, k, dims=(-2, -1))
        if target["masks"].numel():
            masks = torch.rot90(target["masks"], k, dims=(-2, -1)).contiguous()
            target["masks"] = masks
            target["boxes"] = masks_to_boxes(masks).to(target["boxes"].dtype)
        return img, target


class RandomColorJitter:
    """Variação de brilho/contraste/saturação/matiz (só na imagem)."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1).item() < self.p:
            img = self.jitter(img)
        return img, target
