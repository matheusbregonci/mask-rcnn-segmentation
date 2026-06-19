"""
Granulômetro - distribuição de tamanho e volume dos grânulos a partir das segmentações.

Lê o `results.json` gerado pelo predict.py (com os polígonos de cada grânulo) e plota:
  1. Histograma de TAMANHO (diâmetro equivalente ou área);
  2. Histograma de VOLUME estimado por sólido de revolução (rotação 2π do
     polígono em torno do seu eixo maior), salvo em px³ E em mm³.

Com a calibração pixel/milímetro (--px-per-mm), os tamanhos saem em milímetros.

Uso:
    python granulometer.py --source runs/predict/exp/results.json
    python granulometer.py --source runs/predict/exp/results.json --px-per-mm 11.8
    python granulometer.py --source runs/predict/exp/results.json --metric area
    python granulometer.py --source runs/predict/                 # varre vários results.json
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Granulômetro a partir das segmentações")

    parser.add_argument("--source", required=True,
                        help="results.json, pasta (varre **/results.json) ou padrão glob")
    parser.add_argument("--px-per-mm", type=float, default=11.8,
                        help="Calibração: pixels por milímetro. Sem isso, mede em pixels")
    parser.add_argument("--metric", default="diameter", choices=["diameter", "area"],
                        help="Tamanho a medir: diâmetro equivalente (padrão) ou área")
    parser.add_argument("--bins", type=int, default=30, help="Número de barras do histograma")
    parser.add_argument("--min-area-px", type=float, default=0.0,
                        help="Ignora polígonos com área (px) menor que isso, p/ remover ruído")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Confiança mínima para incluir o grânulo")
    parser.add_argument("--out", default=None,
                        help="PNG do histograma de tamanho (padrão: granulometria.png ao lado do source). "
                             "Os plots de volume são salvos na mesma pasta")

    return parser.parse_args()


def gather_results(source):
    """Resolve --source em uma lista de arquivos results.json."""
    p = Path(source)
    if p.is_dir():
        files = sorted(p.glob("**/results.json"))
    elif any(ch in str(source) for ch in "*?["):
        files = sorted(Path().glob(source))
    else:
        files = [p]
    if not files:
        raise FileNotFoundError(f"Nenhum results.json encontrado em: {source}")
    return files


def polygon_area_px(poly):
    """Área do polígono em pixels (fórmula do cadarço / shoelace)."""
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def revolution_volume_px(poly):
    """Volume estimado girando a silhueta do grânulo 2π em torno do eixo maior.

    Método dos discos: rasteriza o polígono, alinha o eixo maior na horizontal
    (via PCA) e integra π·r(x)² ao longo do eixo, com r(x) = metade da espessura
    da silhueta na posição x. Assume seção transversal circular (simetria axial).
    Retorna o volume em px³.
    """
    pts = np.asarray(poly, dtype=np.float32)
    if len(pts) < 3:
        return 0.0

    # Rasteriza a silhueta numa máscara do tamanho da bounding box
    origin = pts.min(axis=0)
    shifted = np.round(pts - origin).astype(np.int32)
    w = int(shifted[:, 0].max()) + 2
    h = int(shifted[:, 1].max()) + 2
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [shifted], 1)

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0

    # Eixo principal (maior) via PCA dos pixels da silhueta
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = coords.mean(axis=0)
    cov = np.cov((coords - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    angle = np.degrees(np.arctan2(major[1], major[0]))

    # Rotaciona para deixar o eixo maior na horizontal (canvas expandido p/ não cortar)
    diag = int(np.ceil(np.hypot(w, h))) + 2
    M = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), angle, 1.0)
    M[0, 2] += (diag - w) / 2.0
    M[1, 2] += (diag - h) / 2.0
    rot = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_NEAREST)

    ys2, xs2 = np.nonzero(rot)
    if xs2.size == 0:
        return 0.0
    if (ys2.max() - ys2.min()) > (xs2.max() - xs2.min()):
        rot = np.rot90(rot)  # garante integração ao longo do eixo maior

    # Discos: r(x) = metade da espessura vertical da silhueta na coluna x
    thickness = rot.sum(axis=0).astype(np.float64)
    radius = thickness / 2.0
    return float(np.pi * np.sum(radius ** 2))  # Σ π r² · Δx(=1px) -> px³


def plot_hist(values, unit, xlabel, title, out_path, bins):
    """Plota o histograma com D10/D50/D90 e salva. Retorna (d10, d50, d90)."""
    d10, d50, d90 = np.percentile(values, [10, 50, 90])
    stats = (
        f"n = {values.size}\n"
        f"média = {values.mean():.3f} {unit}\n"
        f"mediana = {d50:.3f} {unit}\n"
        f"D10 = {d10:.3f} {unit}\n"
        f"D50 = {d50:.3f} {unit}\n"
        f"D90 = {d90:.3f} {unit}"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)
    for d, color, lbl in [(d10, "tab:green", "D10"), (d50, "tab:orange", "D50"), (d90, "tab:red", "D90")]:
        ax.axvline(d, color=color, linestyle="--", linewidth=1.6, label=f"{lbl} = {d:.2f} {unit}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Contagem de grânulos")
    ax.set_title(title)
    ax.legend()
    ax.text(0.98, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    print(f"  {title}")
    print("    " + stats.replace("\n", " | "))
    print(f"    -> {out_path}")
    return d10, d50, d90


def main():
    args = parse_args()
    files = gather_results(args.source)

    # Lê os polígonos: área (px) e volume de revolução (px³) por grânulo
    areas_px, volumes_px = [], []
    for f in files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for img in data.get("images", []):
            for inst in img.get("instances", []):
                if inst.get("score", 1.0) < args.min_score:
                    continue
                poly = inst["polygon"]
                area = polygon_area_px(poly)
                if area < args.min_area_px or area <= 0:
                    continue
                areas_px.append(area)
                volumes_px.append(revolution_volume_px(poly))

    areas_px = np.asarray(areas_px, dtype=float)
    volumes_px = np.asarray(volumes_px, dtype=float)
    if areas_px.size == 0:
        print("Nenhum grânulo encontrado com os filtros atuais.")
        return

    use_mm = args.px_per_mm is not None
    unit = "mm" if use_mm else "px"
    base = Path(args.out).parent if args.out else Path(files[0]).parent

    # ---------- 1) TAMANHO ----------
    diam_px = 2.0 * np.sqrt(areas_px / np.pi)
    if args.metric == "diameter":
        size_vals = diam_px / args.px_per_mm if use_mm else diam_px
        size_xlabel = f"Diâmetro equivalente ({unit})"
    else:  # area
        if use_mm:
            size_vals = areas_px / (args.px_per_mm ** 2)
            size_xlabel = "Área (mm²)"
        else:
            size_vals = areas_px
            size_xlabel = "Área (px²)"

    size_out = Path(args.out) if args.out else base / "granulometria.png"
    print("Granulometria — TAMANHO:")
    plot_hist(size_vals, unit, size_xlabel,
              "Granulometria" + ("" if use_mm else "  (em pixels)"), size_out, args.bins)

    # ---------- 2) VOLUME (sólido de revolução) ----------
    print("\nGranulometria — VOLUME (solido de revolucao, rotacao 2*pi no eixo maior):")
    plot_hist(volumes_px, "px³", "Volume (px³)",
              "Volume por revolução (px³)", base / "volume_px.png", args.bins)
    if use_mm:
        volumes_mm = volumes_px / (args.px_per_mm ** 3)
        plot_hist(volumes_mm, "mm³", "Volume (mm³)",
                  "Volume por revolução (mm³)", base / "volume_mm.png", args.bins)
    else:
        print("  (passe --px-per-mm para gerar também o volume em mm³)")

    print(f"\nGrânulos analisados: {areas_px.size}")


if __name__ == "__main__":
    main()
