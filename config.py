import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATA_PREPROCESSING = ROOT / "Data_Preprocessing"
MODELING = ROOT / "Modeling"
BIOLOGY = ROOT / "Biological_Interpretation"
MODEL_PK = ROOT / "Model_pk"
DATA_DIR = ROOT / "Data"


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default


ERV_ROOT = env_path("HERV_DATA_ROOT", Path("/data2/shihong/fengmh/data/FMH/ERV"))
DATASET = env_path("HERV_DATASET_DIR", ERV_ROOT / "Dataset")
EXTERNAL_MODEL_ROOT = env_path(
    "HERV_EXTERNAL_MODEL_ROOT", ERV_ROOT / "HERV_MEF_9_21" / "Model_pk"
)
PRETRAINED_WEIGHT_ROOT = env_path(
    "HERV_PRETRAINED_WEIGHT_ROOT", ERV_ROOT / "Model_Comparison"
)

CADUCEUS_WEIGHTS = PRETRAINED_WEIGHT_ROOT / "Caduceus_Ph" / "pretrained_weights"
DNABERT2_WEIGHTS = PRETRAINED_WEIGHT_ROOT / "DNABERT_2" / "pretrained_weights"
HYENADNA_WEIGHTS = PRETRAINED_WEIGHT_ROOT / "HyenaDNA" / "pretrained_weights"

EMBEDDING_CACHE_ROOT = env_path(
    "HERV_EMBEDDING_CACHE_ROOT", EXTERNAL_MODEL_ROOT
)

CADUCEUS_CACHE = (
    EMBEDDING_CACHE_ROOT / "Caduceus_Ph" / "output"
    / "01_caduceus_ph_embedding_cache"
    / "caduceus_ph_forward_rc_embeddings_cache.pt"
)
DNABERT2_CACHE = (
    EMBEDDING_CACHE_ROOT / "DNABERT_2" / "output" / "01_dnabert2_embedding_cache"
    / "dnabert2_embeddings_cache.pt"
)
HYENADNA_CACHE = (
    EMBEDDING_CACHE_ROOT / "HyenaDNA" / "output" / "01_hyenadna_embedding_cache"
    / "hyenadna_embeddings_cache.pt"
)

CHROMENHANCER_MODEL = (
    EXTERNAL_MODEL_ROOT / "ChromEnhancer" / "ChromEnhancer-master" / "util"
    / "models" / "ChrNet1.py"
)
HEAP_MODEL = EXTERNAL_MODEL_ROOT / "HEAP" / "HEAP-main" / "HEAP_W_binary_joint.py"
EPCOT_REPO = EXTERNAL_MODEL_ROOT / "EPCOT" / "EPCOT-main"
EPCOT_CHECKPOINT = EXTERNAL_MODEL_ROOT / "EPCOT" / "models" / "pretrain_dnase.pt"
EPINFORMER_REPO = EXTERNAL_MODEL_ROOT / "EPInformer" / "EPInformer-main"
UNIVERSALEPI_REPO = EXTERNAL_MODEL_ROOT / "UniversalEPI" / "UniversalEPI-main"
UNIVERSALEPI_CHECKPOINT = (
    EXTERNAL_MODEL_ROOT / "UniversalEPI" / "stage1_model_gm12878_k562.pth"
)

EXTERNAL_MODELS = {
    "ChromEnhancer": {
        "repository": "https://github.com/EpiGenomicsCode/Adversarial_Observation",
        "entry_point": CHROMENHANCER_MODEL,
        "weights": "none, model architecture is trained from scratch",
    },
    "HEAP": {
        "repository": "https://github.com/Genentech/HEAP",
        "entry_point": HEAP_MODEL,
        "weights": "none, model architecture is trained from scratch",
    },
    "EPCOT": {
        "repository": "https://github.com/liu-bioinfo-lab/EPCOT",
        "entry_point": EPCOT_REPO,
        "weights": EPCOT_CHECKPOINT,
    },
    "EPInformer": {
        "repository": "https://github.com/pinellolab/EPInformer",
        "entry_point": EPINFORMER_REPO,
        "weights": "https://huggingface.co/JiecongLin/EPInformer",
    },
    "UniversalEPI": {
        "repository": "https://github.com/BoevaLab/UniversalEPI",
        "entry_point": UNIVERSALEPI_REPO,
        "weights": UNIVERSALEPI_CHECKPOINT,
    },
    "Caduceus-Ph": {
        "repository": "https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        "entry_point": CADUCEUS_WEIGHTS,
        "weights": CADUCEUS_WEIGHTS,
    },
    "DNABERT-2": {
        "repository": "https://huggingface.co/zhihan1996/DNABERT-2-117M",
        "entry_point": DNABERT2_WEIGHTS,
        "weights": DNABERT2_WEIGHTS,
    },
    "HyenaDNA": {
        "repository": "https://huggingface.co/long-context-lab/hyena_dna",
        "entry_point": HYENADNA_WEIGHTS,
        "weights": HYENADNA_WEIGHTS,
    },
}

BEDTOOLS = os.environ.get("HERV_BEDTOOLS", "bedtools")

LUAD_TRANSCRIPTOMICS = DATASET / "Telescope" / "LUAD" / "merged"

CUDA_VISIBLE_DEVICES = os.environ.get("HERV_CUDA_VISIBLE_DEVICES", "0")
CUBLAS_WORKSPACE_CONFIG = os.environ.get("HERV_CUBLAS_WORKSPACE_CONFIG", ":4096:8")
DEVICE = "cuda:0"
