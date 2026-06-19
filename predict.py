"""
Inferência com Mask R-CNN (torchvision) - Segmentação de Instâncias

Uso:
    python predict.py --source imagem.jpg
    python predict.py --source pasta/imagens/ --conf 0.5
    python predict.py --source video.mp4
    python predict.py --source pasta_de_pastas/   # 1 análise por subpasta (1 nível)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from mrcnn_seg.model import load_model
from mrcnn_seg.utils import IMG_EXTS, draw_masks, get_run_dir, mask_to_polygon, pick_device

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args():
    parser = argparse.ArgumentParser(description="Inferência Mask R-CNN segmentação")

    parser.add_argument("--weights", default="runs/train/exp-1/weights/best.pt", help="Caminho dos pesos")
    parser.add_argument("--source", required=True, help="Imagem, pasta, pasta de subpastas ou vídeo")
    parser.add_argument("--conf", type=float, default=0.5, help="Confiança mínima (0-1)")
    parser.add_argument("--mask-thresh", type=float, default=0.5, help="Limiar de binarização da máscara")
    parser.add_argument("--device", default="cpu", help="0 (GPU) ou cpu")
    parser.add_argument("--num-classes", type=int, default=None, help="Necessário se o checkpoint não tiver o valor")
    parser.add_argument("--show", action="store_true", help="Exibir resultados em tempo real")
    parser.add_argument("--project", default="runs/predict", help="Diretório de saída")
    parser.add_argument("--name", default="exp", help="Nome do experimento")

    return parser.parse_args()


def list_images(directory):
    """Imagens diretamente dentro de `directory` (não recursivo)."""
    return sorted(f for f in Path(directory).iterdir() if f.suffix.lower() in IMG_EXTS)


def build_jobs(source, out_root):
    """Monta a lista de trabalhos (out_dir, images, video).

    - arquivo de imagem/vídeo -> 1 job na raiz out_root
    - pasta só com imagens    -> 1 job na raiz out_root
    - pasta de subpastas       -> 1 job por subpasta (1 nível), em out_root/<subpasta>
    - padrão glob              -> 1 job na raiz out_root
    """
    p = Path(source)
    jobs = []
    if p.is_dir():
        direct = list_images(p)
        subdirs = [d for d in sorted(p.iterdir()) if d.is_dir() and list_images(d)]
        if subdirs:
            for sd in subdirs:
                jobs.append((out_root / sd.name, list_images(sd), None))
            if direct:  # imagens soltas na raiz da pasta também são processadas
                jobs.append((out_root, direct, None))
        else:
            jobs.append((out_root, direct, None))
    elif p.is_file():
        if p.suffix.lower() in VIDEO_EXTS:
            jobs.append((out_root, [], p))
        else:
            jobs.append((out_root, [p], None))
    else:  # padrão glob
        jobs.append((out_root, sorted(Path().glob(source)), None))
    return jobs


@torch.no_grad()
def infer(model, image_rgb, device, conf, mask_thresh):
    """Roda o modelo em uma imagem RGB (np.array) e devolve (masks, labels, scores)."""
    tensor = F.to_tensor(Image.fromarray(image_rgb)).to(device)
    pred = model([tensor])[0]

    keep = pred["scores"] >= conf
    masks = pred["masks"][keep]            # [N, 1, H, W] probabilidades
    masks = (masks.squeeze(1) > mask_thresh).cpu().numpy()
    labels = pred["labels"][keep].cpu().numpy()
    scores = pred["scores"][keep].cpu().numpy()
    return masks, labels, scores


def process_job(model, device, args, images, video, out_dir):
    """Processa um grupo (imagens e/ou vídeo) e salva imagens anotadas + results.json
    em `out_dir`. Retorna (total_objetos, caminho_do_results_json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    results = {"weights": str(args.weights), "conf": args.conf, "images": []}

    def collect(name, rgb_shape, masks, labels, scores):
        """Monta a entrada de resultados (polígonos + área + score) de uma imagem/frame."""
        h, w = rgb_shape[0], rgb_shape[1]
        entry = {"image": name, "width": int(w), "height": int(h), "instances": []}
        for j in range(len(masks)):
            poly, area_px = mask_to_polygon(masks[j])
            if poly is None:
                continue
            entry["instances"].append({
                "label": int(labels[j]),
                "score": float(scores[j]),
                "area_px": area_px,
                "polygon": poly,
            })
        results["images"].append(entry)

    # --- imagens ---
    for path in images:
        rgb = np.array(Image.open(path).convert("RGB"))
        masks, labels, scores = infer(model, rgb, device, args.conf, args.mask_thresh)
        total += len(masks)
        collect(path.name, rgb.shape, masks, labels, scores)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        plotted = draw_masks(bgr, masks, labels, scores, conf_floor=args.conf)

        if args.show:
            cv2.imshow("Segmentação", plotted)
            cv2.waitKey(1)
        cv2.imwrite(str(out_dir / path.name), plotted)

    # --- vídeo ---
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(out_dir / f"{video.stem}_seg.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks, labels, scores = infer(model, rgb, device, args.conf, args.mask_thresh)
            total += len(masks)
            collect(f"{video.stem}_frame_{frame_idx:06d}", rgb.shape, masks, labels, scores)
            plotted = draw_masks(frame, masks, labels, scores, conf_floor=args.conf)
            writer.write(plotted)
            if args.show:
                cv2.imshow("Segmentação", plotted)
                cv2.waitKey(1)
            frame_idx += 1
        cap.release()
        writer.release()

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return total, results_path


def main():
    args = parse_args()
    device = pick_device(args.device)

    model, _names = load_model(args.weights, device, num_classes=args.num_classes)

    out_root = get_run_dir(args.project, args.name)
    jobs = build_jobs(args.source, out_root)

    grand_total = 0
    for out_dir, images, video in jobs:
        if not images and video is None:
            continue
        total, results_path = process_job(model, device, args, images, video, out_dir)
        grand_total += total
        label = "(raiz)" if out_dir == out_root else out_dir.name
        n = len(images) + (1 if video is not None else 0)
        print(f"[{label}] {total} objetos em {n} fonte(s) -> {results_path}")

    if args.show:
        cv2.destroyAllWindows()

    print(f"\nTotal geral de objetos detectados: {grand_total}")
    print(f"Resultados salvos em: {out_root}/")


if __name__ == "__main__":
    main()
