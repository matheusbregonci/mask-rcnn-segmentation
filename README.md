# mask-rcnn-segmentation

Projeto de **Mask R-CNN** para segmentação de instâncias, análogo ao
`yolov8-segmentation`: mesma estrutura de scripts (`train`, `predict`,
`validate`, `export`) e o **mesmo formato de dataset** (anotações YOLO-seg).

A implementação usa o **Mask R-CNN do torchvision** (`maskrcnn_resnet50_fpn`)
— mesma arquitetura do [matterport/Mask_RCNN](https://github.com/matterport/mask_rcnn),
porém em PyTorch moderno e mantido (o repositório original depende de
TensorFlow 1.x / Keras 2.x e não roda em ambientes atuais).

## Estrutura

```
mask-rcnn-segmentation/
├── config/dataset.yaml      # classes e caminhos (formato igual ao YOLO)
├── data/                    # train/ val/ test/  ->  images/ + labels/
├── mrcnn_seg/               # pacote interno
│   ├── dataset.py           # leitura YOLO-seg -> targets do torchvision
│   ├── model.py             # build_model / load_model
│   ├── engine.py            # loop de treino e loss de validação
│   └── utils.py             # config, dispositivo, diretórios, desenho
├── train.py
├── predict.py
├── validate.py
├── export.py
└── requirements.txt
```

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Instale o torch conforme sua CUDA: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Dataset

Mesmo formato do projeto YOLO — polígonos normalizados em `.txt`
(`class_id x1 y1 x2 y2 ... xn yn`). Coloque os arquivos em:

```
data/<split>/images/*.jpg
data/<split>/labels/*.txt
```

Para **reaproveitar os dados do `yolov8-segmentation`**, edite
`config/dataset.yaml` e aponte:

```yaml
path: ../../yolov8-segmentation/data
```

> Diferença em relação ao YOLO: o torchvision reserva a classe `0` para o
> *background*, então a classe `0` do seu rótulo vira `1` internamente. O
> `nc` no `dataset.yaml` continua sendo o número de classes **reais** — o
> background é adicionado automaticamente.

## Uso

```powershell
# Treino (pesos COCO pré-treinados por padrão)
python train.py --epochs 50 --batch 2 --device 0

# Inferência (renderiza apenas as máscaras, como no projeto YOLO)
python predict.py --weights runs/train/exp/weights/best.pt --source data/test/images --conf 0.5

# Avaliação (mAP de box e de máscara, padrão COCO)
python validate.py --weights runs/train/exp/weights/best.pt --split test

# Exportação
python export.py --weights runs/train/exp/weights/best.pt --format onnx
```

Os checkpoints (`best.pt` / `last.pt`) guardam `num_classes` e os nomes das
classes, então `predict`, `validate` e `export` reconstroem o modelo
automaticamente.

## Argumentos dos scripts

### `train.py`
| Argumento | Padrão | Descrição |
|---|---|---|
| `--data` | `config/dataset.yaml` | Configuração do dataset |
| `--epochs` | `100` | Número de épocas |
| `--batch` | `2` | Tamanho do batch |
| `--lr` | `0.005` | Learning rate |
| `--scheduler` | `cosine` | Agendador de LR (`cosine` / `step`) |
| `--warmup-epochs` | `3` | Épocas de warmup linear (modo cosseno) |
| `--lr-step` | `30` | Período do decaimento (modo step) |
| `--no-augment` | *(flag)* | Desliga augmentação (mantém só flip horizontal) |
| `--workers` | `4` | Workers do DataLoader |
| `--device` | `0` | `0` (GPU), `cuda:1` ou `cpu` |
| `--trainable-layers` | `3` | Camadas do backbone a treinar (0-5) |
| `--no-pretrained` | *(flag)* | Não usar pesos COCO pré-treinados |
| `--amp` | *(flag)* | Mixed precision (apenas GPU) |
| `--project` | `runs/train` | Diretório de saída |
| `--name` | `exp` | Nome do experimento |

### `predict.py`
| Argumento | Padrão | Descrição |
|---|---|---|
| `--weights` | `runs/train/exp-1/weights/best.pt` | Caminho dos pesos |
| `--source` | *(obrigatório)* | Imagem, pasta, pasta de subpastas ou vídeo |
| `--conf` | `0.5` | Confiança mínima (0-1) |
| `--mask-thresh` | `0.5` | Limiar de binarização da máscara |
| `--device` | `cpu` | `0` (GPU) ou `cpu` |
| `--num-classes` | `None` | Necessário se o checkpoint não tiver o valor |
| `--show` | *(flag)* | Exibir resultados em tempo real |
| `--project` | `runs/predict` | Diretório de saída |
| `--name` | `exp` | Nome do experimento |

> Numa pasta de subpastas, gera uma análise por subpasta (1 nível) em
> `runs/predict/<name>/<subpasta>/`, cada uma com imagens anotadas + `results.json`.

### `validate.py`
| Argumento | Padrão | Descrição |
|---|---|---|
| `--weights` | `runs/train/exp/weights/best.pt` | Caminho dos pesos |
| `--data` | `config/dataset.yaml` | Configuração do dataset |
| `--split` | `val` | Split a avaliar (`train` / `val` / `test`) |
| `--conf` | `0.05` | Confiança mínima |
| `--mask-thresh` | `0.5` | Limiar de binarização da máscara |
| `--workers` | `4` | Workers do DataLoader |
| `--device` | `0` | `0` (GPU) ou `cpu` |
| `--num-classes` | `None` | Necessário se o checkpoint não tiver o valor |

### `export.py`
| Argumento | Padrão | Descrição |
|---|---|---|
| `--weights` | `runs/train/exp/weights/best.pt` | Caminho dos pesos |
| `--format` | `onnx` | `onnx` / `torchscript` |
| `--imgsz` | `800` | Lado da imagem dummy para o trace |
| `--opset` | `11` | Opset ONNX |
| `--device` | `cpu` | `0` (GPU) ou `cpu` |
| `--num-classes` | `None` | Necessário se o checkpoint não tiver o valor |

### `granulometer.py`
Granulômetro: lê o `results.json` do `predict.py` e plota histogramas de tamanho
(diâmetro equivalente / área) e de volume (sólido de revolução), em px e em mm.

| Argumento | Padrão | Descrição |
|---|---|---|
| `--source` | *(obrigatório)* | `results.json`, pasta (varre `**/results.json`) ou glob |
| `--px-per-mm` | `11.8` | Calibração: pixels por milímetro |
| `--metric` | `diameter` | Tamanho a medir (`diameter` / `area`) |
| `--bins` | `30` | Número de barras do histograma |
| `--min-area-px` | `0.0` | Ignora polígonos com área (px) menor que isso |
| `--min-score` | `0.0` | Confiança mínima para incluir o grânulo |
| `--out` | `None` | PNG do histograma de tamanho (volume vai na mesma pasta) |

## Notas

- **Batch pequeno**: Mask R-CNN consome bastante VRAM. Comece com `--batch 2`
  e aumente conforme a GPU. Sem GPU, use `--device cpu` (lento).
- **Mixed precision**: `--amp` reduz uso de memória em GPU.
- **`validate.py`** depende de `pycocotools` (já no `requirements.txt`).
- **Export ONNX** do Mask R-CNN é sensível à versão; se falhar, use
  `--format torchscript`.
