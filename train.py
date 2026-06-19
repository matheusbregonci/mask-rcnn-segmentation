"""
Treinamento Mask R-CNN (torchvision) - Segmentação de Instâncias

Uso:
    python train.py
    python train.py --epochs 50 --batch 4 --lr 0.005
    python train.py --device cpu
"""

import argparse

import torch
from torch.utils.data import DataLoader

from mrcnn_seg.dataset import (
    Compose,
    RandomColorJitter,
    RandomHorizontalFlip,
    RandomRotation90,
    RandomVerticalFlip,
    YoloSegDataset,
    collate_fn,
)
from mrcnn_seg.engine import evaluate_loss, train_one_epoch
from mrcnn_seg.model import build_model
from mrcnn_seg.utils import (
    get_run_dir,
    load_dataset_config,
    pick_device,
    split_images_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Treinar Mask R-CNN para segmentação")

    parser.add_argument("--data",             default="config/dataset.yaml")
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--batch",            type=int,   default=2)
    parser.add_argument("--lr",               type=float, default=0.005)
    parser.add_argument("--scheduler",        default="cosine", choices=["cosine", "step"], help="Agendador de LR")
    parser.add_argument("--warmup-epochs",    type=int,   default=3, help="Épocas de warmup linear (modo cosseno)")
    parser.add_argument("--lr-step",          type=int,   default=30, help="Período do decaimento (modo step)")
    parser.add_argument("--no-augment",       action="store_true", help="Desliga augmentação (mantém só flip horizontal)")
    parser.add_argument("--workers",          type=int,   default=4)
    parser.add_argument("--device",           default="0", help="0 (GPU), cuda:1 ou cpu")
    parser.add_argument("--trainable-layers", type=int,   default=3, help="Camadas do backbone a treinar (0-5)")
    parser.add_argument("--no-pretrained",    action="store_true", help="Não usar pesos COCO pré-treinados")
    parser.add_argument("--amp",              action="store_true", help="Mixed precision (apenas GPU)")
    parser.add_argument("--project",          default="runs/train")
    parser.add_argument("--name",             default="exp")

    return parser.parse_args()


def build_train_transforms(args):
    """Augmentação de treino. Para grânulos: flips + rotações de 90°
    (objeto sem orientação canônica) + jitter de cor (robustez a iluminação)."""
    if args.no_augment:
        return Compose([RandomHorizontalFlip(0.5)])
    return Compose([
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        RandomRotation90(),
        RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    ])


def build_scheduler(optimizer, args):
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # Cosseno (com warmup linear opcional)
    eta_min = args.lr * 0.01
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta_min)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup, eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])


def main():
    args = parse_args()

    cfg = load_dataset_config(args.data)
    num_classes = cfg["_nc"] + 1  # +1: background
    device = pick_device(args.device)

    train_tf = build_train_transforms(args)
    train_ds = YoloSegDataset(split_images_dir(cfg, "train"), transforms=train_tf)
    val_split = "val" if "val" in cfg else "train"
    val_ds = YoloSegDataset(split_images_dir(cfg, val_split))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn,
    )

    model = build_model(
        num_classes,
        pretrained=not args.no_pretrained,
        trainable_backbone_layers=args.trainable_layers,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    out_dir = get_run_dir(args.project, args.name)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nClasses:     {cfg['_names']} (+ background)")
    print(f"Dataset:     {cfg['_base']}")
    print(f"Treino/Val:  {len(train_ds)} / {len(val_ds)} imagens")
    print(f"Épocas:      {args.epochs}")
    print(f"Batch:       {args.batch}")
    print(f"LR:          {args.lr} ({args.scheduler}" + (f", warmup={args.warmup_epochs}" if args.scheduler == 'cosine' else f", step={args.lr_step}") + ")")
    print(f"Augmentação: {'desligada (só hflip)' if args.no_augment else 'hflip+vflip+rot90+colorjitter'}")
    print(f"Dispositivo: {device}")
    print(f"Saída:       {out_dir}\n")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Época {epoch:>3}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.2e}")

        ckpt = {
            "model": model.state_dict(),
            "num_classes": num_classes,
            "names": cfg["_names"],
            "epoch": epoch,
        }
        torch.save(ckpt, weights_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, weights_dir / "best.pt")
            print(f"         novo melhor (val_loss={best:.4f}) salvo em weights/best.pt")

    print(f"\nTreinamento concluído. Pesos em: {weights_dir}")


if __name__ == "__main__":
    main()
