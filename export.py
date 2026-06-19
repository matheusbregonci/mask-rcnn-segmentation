"""
Exportar o Mask R-CNN para deploy (ONNX ou TorchScript).

Uso:
    python export.py --format onnx
    python export.py --format torchscript

Observação: a exportação para ONNX do Mask R-CNN é sensível à versão do
torch/onnx. Use opset >= 11. Se falhar, o TorchScript é a alternativa mais
estável para deploy em PyTorch.
"""

import argparse
from pathlib import Path

import torch

from mrcnn_seg.model import load_model
from mrcnn_seg.utils import pick_device

FORMATS = ["onnx", "torchscript"]


def parse_args():
    parser = argparse.ArgumentParser(description="Exportar Mask R-CNN")

    parser.add_argument("--weights", default="runs/train/exp/weights/best.pt", help="Caminho dos pesos")
    parser.add_argument("--format", default="onnx", choices=FORMATS, help="Formato de exportação")
    parser.add_argument("--imgsz", type=int, default=800, help="Lado da imagem dummy para o trace")
    parser.add_argument("--opset", type=int, default=11, help="Opset ONNX")
    parser.add_argument("--device", default="cpu", help="0 (GPU) ou cpu")
    parser.add_argument("--num-classes", type=int, default=None, help="Necessário se o checkpoint não tiver o valor")

    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

    model, _names = load_model(args.weights, device, num_classes=args.num_classes)
    model.eval()

    out_base = Path(args.weights).with_suffix("")
    dummy = torch.rand(3, args.imgsz, args.imgsz, device=device)

    if args.format == "onnx":
        out_path = out_base.with_suffix(".onnx")
        torch.onnx.export(
            model,
            ([dummy],),
            str(out_path),
            opset_version=args.opset,
            input_names=["images"],
            output_names=["boxes", "labels", "scores", "masks"],
            dynamic_axes={
                "images": {1: "height", 2: "width"},
                "boxes": {0: "num_detections"},
                "labels": {0: "num_detections"},
                "scores": {0: "num_detections"},
                "masks": {0: "num_detections"},
            },
        )
    else:  # torchscript
        out_path = out_base.with_suffix(".torchscript")
        scripted = torch.jit.script(model)
        scripted.save(str(out_path))

    print(f"\nModelo exportado: {out_path}")


if __name__ == "__main__":
    main()
