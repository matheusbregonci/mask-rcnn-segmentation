"""Construção e carregamento do Mask R-CNN (torchvision)."""

import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes, pretrained=True, trainable_backbone_layers=3):
    """Mask R-CNN ResNet50-FPN com cabeças ajustadas para `num_classes`.

    `num_classes` INCLUI o background (ex.: 1 classe real -> num_classes=2).
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Cabeça de classificação/box
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Cabeça de máscara
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def load_model(weights, device, num_classes=None, names=None):
    """Reconstrói o modelo a partir de um checkpoint salvo pelo train.py.

    Aceita tanto o dict {"model": state, "num_classes": ..., "names": ...}
    quanto um state_dict puro (nesse caso `num_classes` é obrigatório).
    """
    # weights_only=False: o checkpoint guarda também num_classes/names (não só tensores).
    # São arquivos gerados localmente pelo train.py, portanto confiáveis.
    ckpt = torch.load(weights, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        num_classes = num_classes or ckpt.get("num_classes")
        names = names or ckpt.get("names")
    else:
        state = ckpt

    if num_classes is None:
        raise ValueError(
            "num_classes não encontrado no checkpoint; informe --num-classes."
        )

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, names
