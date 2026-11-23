# RGB + Thermal AI for Rheumatoid Arthritis (RA)

A lightweight pipeline that reads a paired **RGB + thermal** hand image, crops the finger joints, and scores RA likelihood with an uncertainty estimate. Built to run on edge hardware (NVIDIA Jetson).

> **Status: research prototype.** The public FLIR One Pro hand dataset has **no RA / non-RA labels**, so the classifier here is only wiring-tested with stand-in labels. It is **not** a diagnostic tool, and the score is not the probability that a person has RA.

## Pipeline

| Stage | What it does |
|---|---|
| 01 Capture | Load paired RGB frames and radiometric thermal frames (°C). Frame *i* of each stream is the same moment. |
| 02 Quality gate | Blur, exposure and thermal-contrast checks before any model runs. |
| 03 Align & preprocess | Automatic thermal-to-RGB registration (landmark-based, no calibration board), gamma correction, denoising autoencoder for thermal. |
| 04 Hand & joints | MediaPipe Hands gives 21 landmarks. Crops of 4 MCP and 4 PIP joints (96×96). Second gate: landmark confidence and hand-mask agreement. |
| 05 Features | Two **MobileNetV3-Small** branches, one for RGB (3 ch) and one for thermal (2 ch: absolute temperature and joint-minus-palm ΔT). One 576-d token per joint. |
| 06 Attention | Joint attention within each modality plus cross-modal attention (RGB ↔ thermal). |
| 07 Fusion | `early`, `intermediate` (proposed: attention + learned modality gate), or `decision`. |
| 08 Classifier | MLP head, Monte-Carlo dropout for score ± uncertainty, patient-independent 70/10/20 split. |
| 09 Deploy | Parameter count, latency benchmark, ONNX export, TensorRT FP16 on Jetson. |

## Design targets

| Metric | Target |
|---|---|
| F1 (unseen patients) | > 77.72% |
| AUROC | ≥ 0.90 |
| Parameters | ≤ 5M (about 2M used) |
| Latency per pair | ≤ 50 ms on Jetson |

## Data

- **FLIR One Pro hand dataset:** 17 subjects (S1–S17), right and left hand, `static` and `open_close` tasks.
- Per recording: an RGB `.mp4` (portrait, black side bars cropped) and a radiometric `.npy` array (frames × 640 × 480, °C).
- The notebook uses the **right hand** only. Up to 12 quality-passing frames per recording are kept.
- **Labels:** none for RA. Stand-in labels (static = 0, open_close = 1) are used only to prove the training and evaluation code works. Real results need a labelled RA dataset.

## Quick start (Kaggle)

1. Add the FLIR One Pro hand dataset to the notebook and turn **Internet ON**.
2. Open `ai-project-iith-fixed.ipynb`.
3. Run the first cell (`%pip install -q mediapipe`). If you see a numpy/protobuf error, restart the kernel and run again.
4. **Restart & Run All.**

Running locally: set `ROOT = Path("path/to/FLIROne")` in the setup cell and install:

```bash
pip install numpy opencv-python mediapipe torch torchvision scikit-learn onnx matplotlib
```

## Thermal-to-RGB registration

The two cameras see the hand at different size and position, so RGB landmarks drawn on raw thermal land in the wrong place. Stage 03b fixes this automatically:

- Runs MediaPipe on the RGB frame and on a rendering of the thermal frame.
- Fits scale, shift and a small rotation from the matched landmarks, pooled over many frames with RANSAC.
- Stores the result in `H_REG`. `iter_pairs()` then returns thermal already warped onto the RGB pixel grid.
- Falls back to hand-silhouette matching if MediaPipe cannot find the hand in thermal.

The orientation check fits scale and shift only (no rotation), so flipped or upside-down thermal frames show a large error. For this dataset the raw thermal orientation (`ORIENT = None`) is correct.

## Evaluation

- Subject-level split (a subject never appears in more than one of train, validation, test).
- Decision threshold chosen on **validation** only, then applied unchanged to test.
- Reports F1, AUROC, precision, recall, accuracy, ROC curve and confusion matrix.

## Deploy to Jetson

```bash
trtexec --onnx=ra_model.onnx --fp16 --saveEngine=ra_model.plan
```

Benchmark on the Jetson itself. The notebook latency covers the neural network only. Measure MediaPipe, registration and cropping separately for end-to-end time.

## Repository layout

```
ai-project-iith (3).ipynb   # full pipeline, stages 01-09
README.md
```

## Limitations

- No RA-labelled data in the public dataset, so reported metrics are placeholders.
- Registration is a single global transform; large changes in hand distance add small errors.
- The palm is used as the temperature reference; switch to the dorsum for clinical use.

## Citation / acknowledgements

- FLIR One Pro hand dataset: add the dataset citation and link here.
- MediaPipe Hands (Google) and MobileNetV3 (Howard et al.).

