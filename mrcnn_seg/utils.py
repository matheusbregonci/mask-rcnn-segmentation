"""Funções auxiliares: config do dataset, dispositivo, diretórios e desenho."""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .dataset import IMG_EXTS  # noqa: F401  (reexport por conveniência)


def load_dataset_config(path):
    """Lê o dataset.yaml (mesmo formato do projeto YOLO) e normaliza campos.

    Adiciona chaves auxiliares:
        _base  -> Path absoluto da raiz dos dados (resolvido a partir de `path`),
        _names -> lista de nomes de classe ordenada por índice,
        _nc    -> número de classes reais (sem background).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = (path.parent / cfg.get("path", ".")).resolve()
    cfg["_base"] = base

    names = cfg.get("names", {})
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cfg["_names"] = list(names)
    cfg["_nc"] = int(cfg.get("nc", len(cfg["_names"])))
    return cfg


def split_images_dir(cfg, split):
    """Caminho absoluto da pasta de imagens de um split (train/val/test)."""
    return cfg["_base"] / cfg[split]


def get_run_dir(project, name):
    """Diretório de saída único, incrementando sufixo: exp, exp-1, exp-2, ..."""
    project = Path(project)
    base = project / name
    if not base.exists():
        return base
    i = 1
    while (project / f"{name}-{i}").exists():
        i += 1
    return project / f"{name}-{i}"


def pick_device(device):
    """Resolve string de dispositivo ('0', 'cuda', 'cpu', 'cuda:1') em torch.device."""
    d = str(device).lower()
    if d == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if d in ("cuda", "gpu", ""):
        return torch.device("cuda")
    if d.startswith("cuda:"):
        return torch.device(d)
    if d.isdigit():
        return torch.device(f"cuda:{d}")
    return torch.device("cuda")


def color_for(idx):
    """Cor BGR estável e distinta por índice de classe."""
    rng = np.random.RandomState(int(idx) * 7 + 1)
    return tuple(int(c) for c in rng.randint(60, 256, size=3))


def mask_to_polygon(mask_bool, epsilon_ratio=0.01):
    """Extrai o maior contorno externo da máscara.

    Retorna (polygon, area_px):
        polygon -> lista [[x, y], ...] em pixels (None se a máscara for vazia),
        area_px -> área real da máscara em pixels (soma dos pixels ativos).
    """
    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    cnt = max(contours, key=cv2.contourArea)
    eps = epsilon_ratio * cv2.arcLength(cnt, True)  # simplificação leve
    cnt = cv2.approxPolyDP(cnt, eps, True)
    return cnt.reshape(-1, 2).tolist(), int(mask_bool.sum())


def _mask_centroid(mask_bool):
    """Centróide (cx, cy) da máscara, robusto a formas não convexas."""
    ys, xs = np.where(mask_bool)
    return int(round(xs.mean())), int(round(ys.mean()))


def score_to_color(score, lo=0.0, hi=1.0):
    """Mapeia confiança em cor BGR: vermelho (baixa) -> amarelo -> verde (alta).

    O gradiente é esticado sobre [lo, hi], de modo que `lo` (limiar de
    confiança) fica vermelho e `hi` fica verde, realçando o contraste.
    """
    t = (float(score) - lo) / max(hi - lo, 1e-6)
    t = max(0.0, min(1.0, t))
    hue = int(t * 60)  # escala OpenCV: 0=vermelho, 30=amarelo, 60=verde
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def _put_centered_text(img, text, center, color=(255, 255, 255), scale=0.5, thickness=1):
    """Escreve `text` centralizado em `center`, com contorno preto p/ legibilidade."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    org = (center[0] - tw // 2, center[1] + th // 2)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_masks(image_bgr, masks, labels, scores=None, alpha=0.5, conf_floor=0.0):
    """Sobrepõe as máscaras e, no centro de cada objeto, a confiança.

    A cor da máscara reflete a confiança (vermelho=baixa -> verde=alta,
    esticado sobre [conf_floor, 1]). Sem `scores`, usa cor por classe.
    O texto da confiança é sempre branco.

    masks      -> array bool/0-1 [N, H, W]
    labels     -> array [N] de índices de classe (já com background = 0)
    scores     -> array [N] de confianças (0-1); se None, não colore por conf nem escreve o texto
    conf_floor -> limiar de confiança; estica o gradiente de cor sobre [conf_floor, 1]
    """
    out = image_bgr.copy()
    for i in range(len(masks)):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        color = score_to_color(scores[i], lo=conf_floor) if scores is not None else color_for(labels[i])
        overlay = out.copy()
        overlay[m] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    # Texto (branco) por cima de todas as máscaras, para não ser encoberto por sobreposições
    if scores is not None:
        for i in range(len(masks)):
            m = masks[i].astype(bool)
            if not m.any():
                continue
            _put_centered_text(out, f"{float(scores[i]):.2f}", _mask_centroid(m))
    return out
