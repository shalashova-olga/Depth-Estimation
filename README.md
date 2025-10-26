# Stereo Depth Estimation Study (InStereo2K)

Комплексный проект по оценке глубины, подготовленный в рамках тестового задания Intern DL Researcher.  
Я выполнила все четыре части: запуск foundation‑моделей, обучение облегчённых архитектур, аналитический обзор и оценку уверенности.

---

## Структура репозитория

```
dl-cv-home-test/
├─ notebooks/
│  ├─ instereo2k.ipynb         # обзор датасета и визуализация disparity
│  ├─ task1_base.ipynb         # Depth-Anything + CREStereo inference
│  ├─ task1_sota.ipynb         # DEFOM-Stereo + RAFT-Stereo inference
│  ├─ task2.ipynb              # обучение и сравнение лёгких моделей
│  └─ task4.ipynb              # per-pixel confidence (MC dropout, TTA, photometric)
├─ results/
│  ├─ task1_depth_anything/    # depth карты для 10 сцен (PNG, NPY)
│  ├─ task1_cres/              # disparity CREStereo
│  ├─ task1_defomstereo/       # disparity DEFOM-Stereo + metadata.json
│  └─ task1_raft/              # disparity RAFT-Stereo + визуализации
├─ models/                     # чекпоинты лёгких моделей из task2
├─ utils/                      # вспомогательные скрипты (метрики и т.п.)
├─ download_dataset.sh         # скрипт загрузки подвыборки InStereo2K
├─ requirements.txt            # зафиксированные зависимости (Python 3.10+)
└─ Обзор моделей и датасетов.pdf  # аналитический обзор (task3)
```

---

## Подготовка окружения

```bash
git clone https://github.com/shalashova-olga/dl-cv-home-test.git
cd dl-cv-home-test

conda create -n dlcv python=3.10 -y
conda activate dlcv
pip install -r requirements.txt
```

### Данные

```bash
bash download_dataset.sh    # скачивает подвыборку InStereo2K в data/instereo2k_sample
```

### Внешние репозитории (обязательны для Run All)

Clone в ту же директорию, где лежит основной проект (скрипты ищут их в `../…`):

```bash
cd ..
git clone https://github.com/LiheYoung/Depth-Anything.git
git clone https://github.com/ibaiGorordo/CREStereo-Pytorch.git
git clone https://github.com/autonomousvision/DEFOM-Stereo.git
git clone https://github.com/princeton-vl/RAFT-Stereo.git
cd dl-cv-home-test
```

### Параметры моделей

- **Depth-Anything** — загружается из hub (`DepthAnything.from_pretrained`).
- **CREStereo** — файл `crestereo_eth3d.pth` ожидается в `CREStereo-Pytorch/models/` или `dl-cv-home-test/models/`. Можно взять из [релиза проекта](https://github.com/ibaiGorordo/CREStereo-Pytorch).
- **DEFOM-Stereo** — требуется `checkpoints/defomstereo_vitl_sceneflow.pth` внутри клонированного репозитория (`bash scripts/download_models.sh` в `DEFOM-Stereo`).
- **RAFT-Stereo** — нужен `models/raftstereo-middlebury.pth` (скачивается через `RAFT-Stereo/download_models.sh` или вручную, файл можно положить в `dl-cv-home-test/models/`).

После подготовки дерево папок выглядит так:

```
../
├─ Depth-Anything/
├─ CREStereo-Pytorch/
├─ DEFOM-Stereo/
├─ RAFT-Stereo/
└─ dl-cv-home-test/
```

---

## Ноутбуки и результаты

| Задание | Ноутбук | Что делает | Где лежит вывод |
| --- | --- | --- | --- |
| Task 0 | `instereo2k.ipynb` | Быстрый просмотр подвыборки, disparity heatmaps, проверка структуры данных | без сохранения |
| Task 1 (heavy) | `task1_base.ipynb` | Depth-Anything (mono depth) и CREStereo (stereo) → 10 сцен | `results/task1_depth_anything`, `results/task1_cres` |
| Task 1 (SOTA) | `task1_sota.ipynb` | DEFOM-Stereo + RAFT-Stereo, единый семплер сцен, метаданные и визуализации | `results/task1_defomstereo`, `results/task1_raft` |
| Task 2 | `task2.ipynb` | Обучение нескольких лёгких архитектур, сравнение лоссов и функций штрафа | чекпоинты в `models/`, метрики печатаются в ноутбуке |
| Task 3 | — | Аналитический обзор моделей и датасетов | `Обзор моделей и датасетов.pdf` |
| Task 4 | `task4.ipynb` | Per-pixel confidence: MC Dropout, Test-Time Augmentation, Photometric error | интерактивные визуализации внутри ноутбука |

### Task 1 — инференс foundation‑моделей

- **Depth-Anything ViT-L (mono)**: 10 сцен, PNG + NPY, папка `results/task1_depth_anything/`.
- **CREStereo ETH3D**: 10 сцен, PNG + NPY, `results/task1_cres/`.
- **DEFOM-Stereo ViT-L**: 10 сцен, PNG+NPY и `metadata.json` с диапазонами disparity, `results/task1_defomstereo/`.
- **RAFT-Stereo Middlebury**: 10 сцен, PNG+NPY и коллажи с MAE/RMSE на GT disparity, `results/task1_raft/`.

Все пайплайны используют единый случайный seed (42) и автоматически сохраняют список обработанных сцен для воспроизводимости.

### Task 2 — лёгкие модели

Училась на подвыборке псевдоразметки от DEFOM (80 % train / 20 % val), обучение — в Mixed Precision на M2 Pro.  
Основные результаты на валидации (disparity, px):

| Модель | Лосс | MAE | RMSE |
| --- | --- | --- | --- |
| Stereo U-Net | L1 | 31.77 | 34.68 |
| Mobile StereoNet | L1 | 30.03 | 32.75 |
| Stereo U-Net | SSIM | **4.90** | **6.46** |
| Stereo U-Net | L1 + SSIM | 5.89 | 7.91 |
| AnetNet (Hybrid) | L1 + SSIM | 4.97 | 6.48 |

Лучшие веса сохранены в `models/` (`*_best.pth`) и подгружаются в конце ноутбука для быстрой валидации.

### Task 3 — аналитика

Файл `Обзор моделей и датасетов.pdf` содержит 2× по 8–10 работ (модели и датасеты), выделены ключевые идеи, ограничения и ссылки на оригинальные статьи/лидерборды.

### Task 4 — confidence estimation

`task4.ipynb` реализует три метода оценки уверенности без изменения архитектуры модели:
1. **MC Dropout** — 15 прогонов RAFT-Стерео с активированным dropout.
2. **Test-Time Augmentation** — случайные horizontal flips, агрегация диспаратности.
3. **Photometric Reconstruction Error** — reprojection right→left и L1 ошибка.

Для каждой карты строятся визуализации и сравнительный блок с disparity (см. вывод ячейки 3).

---

## Как воспроизвести

1. Подготовить окружение и загрузить данные (см. выше).
2. Проверить, что внешние репозитории доступны (`Depth-Anything`, `CREStereo-Pytorch`, `DEFOM-Stereo`, `RAFT-Stereo`).
3. Открыть Jupyter Lab/Notebook и выполнить `Run All` для интересующего ноутбука:

```bash
jupyter lab
# или
jupyter nbconvert --to notebook --execute notebooks/task1_sota.ipynb
```

4. Готовые артефакты появятся в `results/` и `models/`.

---

## Контейнеризация (Docker)

Чтобы не разворачивать окружение вручную, можно использовать Docker.

1. **Сборка образа**
   ```bash
   docker build -t dlcv-depth .
   ```

2. **Запуск контейнера**
   ```bash
   docker run --rm -it \
     -p 8888:8888 \
     -v $(pwd)/data:/workspace/data \
     -v $(pwd)/results:/workspace/results \
     dlcv-depth
   ```
   После старта Jupyter Lab доступен на `http://localhost:8888/` (токен пустой).

3. **Данные**  
   Внутри контейнера можно выполнить `bash download_dataset.sh` или заранее положить загруженную подвыборку в локальный `data/` — том уже смонтирован.

4. **Внешние репозитории**  
   Dockerfile автоматически клонирует `Depth-Anything`, `CREStereo-Pytorch`, `DEFOM-Stereo`, `RAFT-Stereo` в `/workspace/external`. При необходимости обновите зависимости и пересоберите образ.

5. **GPU (опционально)**  
   Текущий Dockerfile рассчитан на CPU. Для CUDA можно сменить базовый образ на `nvidia/cuda` и запускать с `--gpus all`.

---

## Примечания

- Ноутбуки автоматически определяют устройство (`cuda` / `mps` / CPU). Для Apple Silicon нужно убедиться, что установлен PyTorch с поддержкой `mps`.
- `results/` хранит только вывод модели. Датасет не коммитится: `data/` добавлена в `.gitignore`.
- Параметры выборки, seed и конфигурации inference вынесены в переменные в начале ноутбуков для удобной модификации.
