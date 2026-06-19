"""Loop de treino e avaliação de loss para o Mask R-CNN."""

import math

import torch


def _to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=10, scaler=None):
    """Uma época de treino. Retorna a loss média da época."""
    model.train()
    running = 0.0
    n = len(loader)

    for i, (images, targets) in enumerate(loader):
        images, targets = _to_device(images, targets, device)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"  [aviso] loss não-finita ({loss_value}); batch ignorado")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        running += loss_value
        if i % print_freq == 0:
            parts = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items())
            print(f"  época {epoch} [{i:>4}/{n}] loss={loss_value:.3f} | {parts}")

    return running / max(1, n)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Loss média no conjunto de validação.

    Modelos de detecção do torchvision só devolvem losses em modo .train();
    usamos no_grad para não acumular gradiente.
    """
    model.train()
    total = 0.0
    n = len(loader)
    for images, targets in loader:
        images, targets = _to_device(images, targets, device)
        loss_dict = model(images, targets)
        total += sum(loss for loss in loss_dict.values()).item()
    return total / max(1, n)
