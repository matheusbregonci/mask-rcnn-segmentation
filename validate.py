"""
Avaliação do Mask R-CNN treinado (mAP de box e de máscara, padrão COCO).

Uso:
    python validate.py
    python validate.py --weights runs/train/exp/weights/best.pt --split test
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from mrcnn_seg.dataset import YoloSegDataset, collate_fn
from mrcnn_seg.model import load_model
from mrcnn_seg.utils import load_dataset_config, pick_device, split_images_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Validar Mask R-CNN segmentação")

    parser.add_argument("--weights", default="runs/train/exp/weights/best.pt", help="Caminho dos pesos")
    parser.add_argument("--data", default="config/dataset.yaml", help="Caminho do dataset.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split a avaliar")
    parser.add_argument("--conf", type=float, default=0.05, help="Confiança mínima para considerar a predição")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Limiar de binarização da máscara")
    parser.add_argument("--workers", type=int, default=4, help="Workers do DataLoader")
    parser.add_argument("--device", default="0", help="0 (GPU) ou cpu")
    parser.add_argument("--num-classes", type=int, default=None, help="Necessário se o checkpoint não tiver o valor")

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = pick_device(args.device)

    cfg = load_dataset_config(args.data)
    model, _names = load_model(args.weights, device, num_classes=args.num_classes)

    ds = YoloSegDataset(split_images_dir(cfg, args.split))
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    # Métricas separadas para box (bbox) e máscara (segm), formato COCO.
    map_box = MeanAveragePrecision(iou_type="bbox")
    map_seg = MeanAveragePrecision(iou_type="segm")

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds = model(images)

        preds_box, preds_seg, t_box, t_seg = [], [], [], []
        for p, t in zip(preds, targets):
            keep = p["scores"] >= args.conf
            masks = (p["masks"][keep].squeeze(1) > args.mask_thresh).to(torch.uint8).cpu()
            labels = p["labels"][keep].cpu()
            scores = p["scores"][keep].cpu()
            boxes = p["boxes"][keep].cpu()

            preds_box.append({"boxes": boxes, "scores": scores, "labels": labels})
            preds_seg.append({"masks": masks.bool(), "scores": scores, "labels": labels})
            t_box.append({"boxes": t["boxes"], "labels": t["labels"]})
            t_seg.append({"masks": t["masks"].bool(), "labels": t["labels"]})

        map_box.update(preds_box, t_box)
        map_seg.update(preds_seg, t_seg)

    res_box = map_box.compute()
    res_seg = map_seg.compute()

    print("\n--- Métricas (padrão COCO) ---")
    print(f"mAP50    (box):  {res_box['map_50'].item():.4f}")
    print(f"mAP50-95 (box):  {res_box['map'].item():.4f}")
    print(f"mAP50    (mask): {res_seg['map_50'].item():.4f}")
    print(f"mAP50-95 (mask): {res_seg['map'].item():.4f}")


if __name__ == "__main__":
    main()
