# Human_endometrial_fertility_histology_DL_study
Code collection for training and testing of fertility outcome-oriented deep learning models based on endometrial HE histology images.

All codes should be well validated and workable following instructions. Nonetheless, please do not hesitate to contact author for any potential bugs or questions.

This repository packages an exploratory endometrium implantation prediction workflow using:
- UNI2-h feature-based MIL models for WSI and LE
- ResNet-18 MIL baselines on raw patches
- Logistic-regression multimodal fusion (WSI + LE + clinical)
- Attention/GradCAM visualization utilities

## Scope and Positioning

This codebase focuses on **usability and integration** of established models and tools in a single, reproducible pipeline for our data setting. It is **not** introducing a novel architecture. The pipeline was assembled and adapted from existing model families/tooling to fit practical requirements (data format, split logic, ensemble evaluation, and visualization).

This work relies heavily on the Trident toolkit workflow for WSI processing. In our setup, UNI2-h features were generated from whole-slide files in `.svs`/`.ndpi` formats following the recommended Trident processing sequence.

## Implemented Model Families

- **UNI2-h feature-based pipelines**
  - These pipelines expect pre-extracted UNI2-h H5 features (`features`, 1536-d):
  - WSI pooling MIL: `PoolingClassifier`
  - WSI ACMIL: `ACMILClassifier`
  - LE pooling MIL: `PoolingClassifier`
- **ResNet-18 patch-image pipelines**
  - These pipelines expect raw RGB patch images per slide folder:
  - WSI patch-level ResNet-18 with slide-level aggregation
  - LE patch-level ResNet-18 with slide-level aggregation
- **Multimodal fusion**
  - 2-factor: WSI prob + LE prob (logistic regression)
  - 4-factor: WSI prob + LE prob + Age + Endometrial thickness
  - Clinical-only: Age + Endometrial thickness

## Folder Layout

```text
github_upload/
  training/
    train_wsi_pooling.py
    train_wsi_acmil.py
    train_wsi_resnet18.py
    train_le_pooling.py
    train_le_resnet18.py
    train_multimodal_2factor.py
    train_multimodal_4factor.py
    train_multimodal_clinical.py
    models.py
  testing/
    test_wsi_pooling.py
    test_wsi_acmil.py
    test_wsi_resnet18.py
    test_le_pooling.py
    test_le_resnet18.py
    test_multimodal_2factor.py
    test_multimodal_4factor.py
    test_multimodal_clinical.py
  gradcam/
    acmil_attention_wsi.py
    gradcam_le_resnet18.py
  checkpoints/
    wsi_pooling/
    wsi_acmil/
    wsi_resnet18/
    le_pooling/
    le_resnet18/
    multimodal_2factor/
    multimodal_4factor/
    multimodal_clinical/
```


## Data Requirements

### 1) Split CSVs (required)

Internal split CSV (for WSI or LE):

```csv
sample_id,split,label
101,training,0
103,testing,1
```

- Required columns: `sample_id`, `split`, `label`
- Typical split values:
  - Internal: `training`, `testing`
  - External: `ext_test` (in external split CSV)

### 2) UNI2-h feature directories (H5)

Expected structure:

```text
<features_dir>/
  <sample_id>.h5
  <sample_id>.h5
```

Each H5 file must contain dataset key:
- `features` with shape `(num_patches, 1536)`

### 3) Raw patch directories (ResNet-18)

Expected structure:

```text
<patch_root>/
  <sample_id>/
    patch_0001.png
    patch_0002.png
  <sample_id>/
    ...
```

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`

### 4) Clinical CSVs (for 4-factor and clinical-only)

```csv
Sample_no,AgeW_EB,EnTh
101,39,10.0
102,31,8.0
```

- Required columns: `Sample_no`, `AgeW_EB`, `EnTh`

### 5) CSV requirements by pipeline

| Pipeline | Required CSV(s) | Required columns | Required split values |
|---|---|---|---|
| WSI UNI2-h pooling (`train/test_wsi_pooling.py`) | internal split CSV + external split CSV | `sample_id, split, label` | internal: `training/testing`; external: `ext_test` |
| WSI UNI2-h ACMIL (`train/test_wsi_acmil.py`) | internal split CSV + external split CSV | `sample_id, split, label` | internal: `training/testing`; external: `ext_test` |
| LE UNI2-h pooling (`train/test_le_pooling.py`) | internal split CSV + external split CSV | `sample_id, split, label` | internal: `training/testing`; external: `ext_test` |
| WSI ResNet-18 (`train/test_wsi_resnet18.py`) | internal split CSV + external split CSV | `sample_id, split, label` | internal: `training/testing`; external: `ext_test` |
| LE ResNet-18 (`train/test_le_resnet18.py`) | internal split CSV + external split CSV | `sample_id, split, label` | internal: `training/testing`; external: `ext_test` |
| Multimodal 2-factor (`train/test_multimodal_2factor.py`) | WSI split CSV + LE split CSV | `sample_id, split, label` | internal: `testing`; external: `ext_test` |
| Multimodal 4-factor (`train/test_multimodal_4factor.py`) | WSI split CSV + LE split CSV + clinical CSV(s) | split CSVs: `sample_id, split, label`; clinical: `Sample_no, AgeW_EB, EnTh` | internal: `testing`; external: `ext_test` |
| Clinical-only (`train/test_multimodal_clinical.py`) | internal dataset CSV + external dataset CSV + clinical CSV(s) | split CSVs: `sample_id, split, label`; clinical: `Sample_no, AgeW_EB, EnTh` | internal: `testing`; external: `ext_test` |

## Environment Setup

Use Python 3.10+ with CUDA-enabled PyTorch where available.

Minimal dependencies:
- `torch`, `torchvision`
- `numpy`, `pandas`, `scikit-learn`
- `h5py`, `matplotlib`, `seaborn`, `Pillow`
- `joblib`
- Visualization extras: `opencv-python`, `grad-cam`, optional `openslide-python`

Example install:

```bash
pip install torch torchvision numpy pandas scikit-learn h5py matplotlib seaborn pillow joblib opencv-python grad-cam
```

## WSI to UNI2-h Features (Trident Workflow)

Example Trident-style processing from our `.svs`/`.ndpi` slides (use your own paths, please refer to the trident package for detailed instructions):

```bash
python run_batch_of_slides.py --task seg --wsi_dir /path/to/project_root/data/wsi_input --job_dir /path/to/project_root/data/patch_job_uni2h --gpu 0 --segmenter hest

python run_batch_of_slides.py --task coords --wsi_dir /path/to/project_root/data/wsi_input --job_dir /path/to/project_root/data/patch_job_uni2h --gpu 0 --mag 20 --patch_size 256 --overlap 0 --min_tissue_proportion 0.35

python run_batch_of_slides.py --task feat --wsi_dir /path/to/project_root/data/wsi_input --job_dir /path/to/project_root/data/patch_job_uni2h --gpu 0 --patch_encoder uni_v2 --patch_size 256 --mag 20
```

After feature extraction, point training/testing scripts to the generated `features_uni_v2` directories.

## Quick Start: Train

Run from `github_upload/`.

### WSI pooling MIL (UNI2-h features)

```bash
python training/train_wsi_pooling.py \
  --train_feats_dir /path/to/wsi/features_train \
  --test_feats_dir /path/to/wsi/features_test \
  --ext_test_feats_dir /path/to/wsi/features_ext \
  --dataset_csv /path/to/wsi_internal_split.csv \
  --ext_test_csv /path/to/wsi_external_split.csv \
  --output_dir outputs/wsi_pooling \
  --head mlp --k_folds 10 --gpu_id 0
```

### WSI ACMIL (UNI2-h features)

```bash
python training/train_wsi_acmil.py \
  --train_feats_dir /path/to/wsi/features_train \
  --test_feats_dir /path/to/wsi/features_test \
  --ext_test_feats_dir /path/to/wsi/features_ext \
  --dataset_csv /path/to/wsi_internal_split.csv \
  --ext_test_csv /path/to/wsi_external_split.csv \
  --output_dir outputs/wsi_acmil \
  --head mlp --k_folds 10 --gpu_id 0
```

### WSI ResNet-18 MIL (raw patches)

```bash
python training/train_wsi_resnet18.py \
  --patches_dir /path/to/wsi_patches_train \
  --test_patches_dir /path/to/wsi_patches_test \
  --ext_patches_dir /path/to/wsi_patches_ext \
  --dataset_csv /path/to/wsi_internal_split.csv \
  --ext_test_csv /path/to/wsi_external_split.csv \
  --output_dir outputs/wsi_resnet18 \
  --k_folds 10 --gpu_id 0
```

### LE pooling MIL (UNI2-h features)

```bash
python training/train_le_pooling.py \
  --train_feats_dir /path/to/le/features_train \
  --test_feats_dir /path/to/le/features_test \
  --ext_test_feats_dir /path/to/le/features_ext \
  --dataset_csv /path/to/le_internal_split.csv \
  --ext_test_csv /path/to/le_external_split.csv \
  --output_dir outputs/le_pooling \
  --k_folds 10 --gpu_id 0
```

### LE ResNet-18 MIL (raw patches)

```bash
python training/train_le_resnet18.py \
  --patches_dir /path/to/le_patches_train \
  --test_patches_dir /path/to/le_patches_test \
  --ext_patches_dir /path/to/le_patches_ext \
  --dataset_csv /path/to/le_internal_split.csv \
  --ext_test_csv /path/to/le_external_split.csv \
  --output_dir outputs/le_resnet18 \
  --k_folds 10 --gpu_id 0
```

### Multimodal fusion

2-factor:

```bash
python training/train_multimodal_2factor.py \
  --wsi_run_dir outputs/wsi_pooling/<run_name> \
  --le_run_dir outputs/le_pooling/<run_name> \
  --output_dir outputs/multimodal_2factor \
  --gpu_id 0
```

4-factor:

```bash
python training/train_multimodal_4factor.py \
  --wsi_run_dir outputs/wsi_pooling/<run_name> \
  --le_run_dir outputs/le_pooling/<run_name> \
  --internal_clinical_csv /path/to/internal_clinical.csv \
  --external_clinical_csv /path/to/external_clinical.csv \
  --output_dir outputs/multimodal_4factor \
  --gpu_id 0
```

Clinical-only:

```bash
python training/train_multimodal_clinical.py \
  --wsi_run_dir outputs/wsi_pooling/<run_name> \
  --internal_clinical_csv /path/to/internal_clinical.csv \
  --external_clinical_csv /path/to/external_clinical.csv \
  --output_dir outputs/multimodal_clinical
```

## Quick Start: Test (Using Your Own Runs)

```bash
python testing/test_wsi_pooling.py --run_dir outputs/wsi_pooling/<run_name> --output_dir outputs_eval/wsi_pooling
python testing/test_wsi_acmil.py --run_dir outputs/wsi_acmil/<run_name> --output_dir outputs_eval/wsi_acmil
python testing/test_wsi_resnet18.py --run_dir outputs/wsi_resnet18/<run_name> --output_dir outputs_eval/wsi_resnet18
python testing/test_le_pooling.py --run_dir outputs/le_pooling/<run_name> --output_dir outputs_eval/le_pooling
python testing/test_le_resnet18.py --run_dir outputs/le_resnet18/<run_name> --output_dir outputs_eval/le_resnet18
```

### Test only one dataset (single-set mode)

Current test scripts evaluate both internal and external sets in one run. Leaving external inputs empty is not supported in the current scripts.

Current supported single-cohort option:

- Reuse the same dataset for both internal/external arguments (same path and same CSV), and define your target cohort with split labels as needed.

Example (LE pooling, single feature set):

```bash
python testing/test_le_pooling.py \
  --run_dir outputs/le_pooling/<run_name> \
  --test_feats_dir /path/to/le_features_single_set \
  --ext_feats_dir /path/to/le_features_single_set \
  --dataset_csv /path/to/single_set_split.csv \
  --ext_test_csv /path/to/single_set_split.csv \
  --output_dir outputs_eval/le_pooling_single_set
```

In `single_set_split.csv`, include rows with `split=testing` and/or `split=ext_test` for the samples you want to evaluate.

Multimodal tests:

```bash
python testing/test_multimodal_2factor.py \
  --wsi_run_dir outputs/wsi_pooling/<run_name> \
  --le_run_dir outputs/le_pooling/<run_name> \
  --logreg_dir outputs/multimodal_2factor/<run_name> \
  --output_dir outputs_eval/multimodal_2factor

python testing/test_multimodal_4factor.py \
  --wsi_run_dir outputs/wsi_pooling/<run_name> \
  --le_run_dir outputs/le_pooling/<run_name> \
  --logreg_dir outputs/multimodal_4factor/<run_name> \
  --internal_clinical_csv /path/to/internal_clinical.csv \
  --external_clinical_csv /path/to/external_clinical.csv \
  --output_dir outputs_eval/multimodal_4factor

python testing/test_multimodal_clinical.py \
  --logreg_dir outputs/multimodal_clinical/<run_name> \
  --internal_clinical_csv /path/to/internal_clinical.csv \
  --external_clinical_csv /path/to/external_clinical.csv \
  --internal_dataset_csv /path/to/wsi_internal_split.csv \
  --external_dataset_csv /path/to/wsi_external_split.csv \
  --output_dir outputs_eval/multimodal_clinical
```

## Quick Start: Test Using Provided Checkpoints

Use the packaged folders under `checkpoints/` directly as run/model inputs.

```bash
python testing/test_wsi_pooling.py --run_dir checkpoints/wsi_pooling --output_dir outputs_eval_ckpt/wsi_pooling
python testing/test_wsi_acmil.py --run_dir checkpoints/wsi_acmil --output_dir outputs_eval_ckpt/wsi_acmil
python testing/test_wsi_resnet18.py --run_dir checkpoints/wsi_resnet18 --output_dir outputs_eval_ckpt/wsi_resnet18
python testing/test_le_pooling.py --run_dir checkpoints/le_pooling --output_dir outputs_eval_ckpt/le_pooling
python testing/test_le_resnet18.py --run_dir checkpoints/le_resnet18 --output_dir outputs_eval_ckpt/le_resnet18
```

For multimodal checkpoint testing, point `--logreg_dir` to:
- `checkpoints/multimodal_2factor`
- `checkpoints/multimodal_4factor`
- `checkpoints/multimodal_clinical`

and point base model run dirs to:
- `checkpoints/wsi_pooling`, `checkpoints/le_pooling`

## Visualization

### ACMIL bag attention on WSI

```bash
python gradcam/acmil_attention_wsi.py \
  --run_dir checkpoints/wsi_acmil \
  --test_feats_dir /path/to/wsi/features_test \
  --wsi_dir /path/to/wsi_files \
  --dataset_csv /path/to/wsi_internal_split.csv \
  --output_dir vis/acmil_attention \
  --fold_id 1 --split testing --gpu_id 0
```

### LE ResNet-18 GradCAM

```bash
python gradcam/gradcam_le_resnet18.py \
  --run_dir checkpoints/le_resnet18 \
  --patches_dir /path/to/le_patches_test \
  --dataset_csv /path/to/le_internal_split.csv \
  --output_dir vis/le_gradcam \
  --fold_id 1 --split testing --gpu_id 0
```

## Example Input Checklist

Before running training/testing, verify:
- CSV has `sample_id, split, label`
- Feature path contains `<sample_id>.h5` with key `features`
- Patch path contains `<sample_id>/` subfolders with image patches
- Clinical CSV has `Sample_no, AgeW_EB, EnTh` (for 4-factor/clinical-only)

## References

This pipeline uses/adapts established components from:
- **UNI / UNI2-h** for feature extraction used by UNI2-h-based LE/WSI MIL pipelines
- **ACMIL** (attention-based multiple-instance learning)
- **ResNet-18** for raw-patch LE/WSI MIL baselines (He et al.)
- **Grad-CAM / pytorch-grad-cam** for visual explanations
- **Trident**-style pathology workflow components for data/model operations

Please cite the original model/tool papers and repositories used in your downstream publication context.
