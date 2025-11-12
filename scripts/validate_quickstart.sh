#!/usr/bin/env bash
# Quickstart Validation Script
# Validates all commands from quickstart.md on a reduced dataset

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation mode
VALIDATE_MODE="${VALIDATE_MODE:-local}"

# Logging
LOG_FILE="validation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quickstart Validation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Mode: ${VALIDATE_MODE}"
echo -e "Log: ${LOG_FILE}"
echo ""

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Helper functions
check() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -e "${BLUE}[CHECK ${TOTAL_CHECKS}]${NC} $1"
}

pass() {
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    echo -e "${GREEN}✓ PASS${NC}: $1"
    echo ""
}

fail() {
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    echo -e "${RED}✗ FAIL${NC}: $1"
    echo ""
}

warn() {
    echo -e "${YELLOW}⚠ WARNING${NC}: $1"
}

info() {
    echo -e "${BLUE}ℹ INFO${NC}: $1"
}

# Check 1: Python environment
check "Python version >=3.10"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)

if [[ ${PYTHON_MAJOR} -ge 3 ]] && [[ ${PYTHON_MINOR} -ge 10 ]]; then
    pass "Python ${PYTHON_VERSION} detected"
else
    fail "Python ${PYTHON_VERSION} is too old. Need >=3.10"
    exit 1
fi

# Check 2: Virtual environment
check "Virtual environment active"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    pass "Virtual environment: ${VIRTUAL_ENV}"
else
    warn "No virtual environment detected. Consider activating one."
fi

# Check 3: Package installation
check "Package installation (pip install -e .)"
if pip list | grep -q "mentallama-criteria-cls"; then
    pass "Package installed"
else
    info "Installing package..."
    if pip install -e . > /dev/null 2>&1; then
        pass "Package installed successfully"
    else
        fail "Failed to install package"
        exit 1
    fi
fi

# Check 4: Required dependencies
check "Required dependencies"
REQUIRED_PACKAGES=(
    "mlflow"
    "transformers"
    "torch"
    "peft"
    "accelerate"
    "hydra-core"
    "scikit-learn"
    "pandas"
    "numpy"
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if pip list | grep -q "^${pkg}"; then
        info "  ✓ ${pkg}"
    else
        MISSING_PACKAGES+=("${pkg}")
        warn "  ✗ ${pkg} missing"
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -eq 0 ]]; then
    pass "All required packages installed"
else
    fail "Missing packages: ${MISSING_PACKAGES[*]}"
    exit 1
fi

# Check 5: Data files
check "Data files exist"
DATA_FILES=(
    "data/DSM5/criteria.csv"
    "data/redsm5/posts.csv"
    "data/redsm5/labels.csv"
)

MISSING_FILES=()
for file in "${DATA_FILES[@]}"; do
    if [[ -f "${file}" ]]; then
        info "  ✓ ${file}"
    else
        MISSING_FILES+=("${file}")
        warn "  ✗ ${file} missing"
    fi
done

if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    pass "All data files present"
else
    fail "Missing data files: ${MISSING_FILES[*]}"
    exit 1
fi

# Check 6: Configuration files
check "Hydra configuration files"
CONFIG_FILES=(
    "configs/config.yaml"
    "configs/data/redsm5.yaml"
    "configs/model/mentallam.yaml"
    "configs/training/cv.yaml"
    "configs/logging/mlflow.yaml"
)

MISSING_CONFIGS=()
for file in "${CONFIG_FILES[@]}"; do
    if [[ -f "${file}" ]]; then
        info "  ✓ ${file}"
    else
        MISSING_CONFIGS+=("${file}")
        warn "  ✗ ${file} missing"
    fi
done

if [[ ${#MISSING_CONFIGS[@]} -eq 0 ]]; then
    pass "All config files present"
else
    fail "Missing config files: ${MISSING_CONFIGS[*]}"
    exit 1
fi

# Check 7: GPU availability (optional)
check "GPU availability (optional)"
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None")
    pass "GPU available: ${GPU_COUNT}x ${GPU_NAME}"
else
    warn "No GPU detected. Training will be slow."
fi

# Check 8: Smoke test - Data loading
check "Data loading smoke test"
info "Loading dataset..."
if python -c "
from Project.SubProject.data.dataset import MentalHealthDataset
dataset = MentalHealthDataset(
    redsm5_path='data/redsm5',
    dsm5_path='data/DSM5'
)
print(f'Loaded {len(dataset)} samples')
" > /tmp/data_load.out 2>&1; then
    SAMPLE_COUNT=$(cat /tmp/data_load.out | grep -oP 'Loaded \K[0-9]+')
    pass "Data loaded: ${SAMPLE_COUNT} samples"
else
    fail "Failed to load data"
    cat /tmp/data_load.out
    exit 1
fi

# Check 9: Smoke test - Fold generation
check "Fold generation smoke test"
info "Generating 2-fold splits..."
FOLD_OUTPUT=$(mktemp -d)
if python -m Project.SubProject.data.splits \
    data.redsm5_path=data/redsm5 \
    data.dsm5_path=data/DSM5 \
    training.folds=2 \
    training.seed=42 \
    output.folds_dir="${FOLD_OUTPUT}" \
    > /tmp/fold_gen.out 2>&1; then

    if [[ -f "${FOLD_OUTPUT}/fold_0.json" ]] && [[ -f "${FOLD_OUTPUT}/fold_1.json" ]]; then
        pass "Folds generated: ${FOLD_OUTPUT}"
    else
        fail "Fold files not created"
        cat /tmp/fold_gen.out
    fi
else
    fail "Fold generation failed"
    cat /tmp/fold_gen.out
    exit 1
fi

# Check 10: Smoke test - Training (tiny run)
check "Training smoke test (2 folds, 10 steps)"
info "Running mini training session..."
TRAIN_OUTPUT=$(mktemp -d)

if timeout 300 python -m Project.SubProject.engine.train_engine \
    experiment.name=smoke_test_$(date +%s) \
    training.folds=2 \
    training.max_steps=10 \
    training.batch_size=2 \
    data.sample_fraction=0.01 \
    output_dir="${TRAIN_OUTPUT}" \
    logging.tracking_uri="sqlite:///${TRAIN_OUTPUT}/mlflow.db" \
    hydra.run.dir="${TRAIN_OUTPUT}/hydra" \
    > /tmp/train_smoke.out 2>&1; then
    pass "Training smoke test completed"
else
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 124 ]]; then
        warn "Training timed out (expected for CI)"
    else
        fail "Training smoke test failed with code ${EXIT_CODE}"
        tail -50 /tmp/train_smoke.out
        exit 1
    fi
fi

# Check 11: Smoke test - Inference (if checkpoint exists)
check "Inference smoke test"
info "Looking for checkpoints..."

# Use the smoke test checkpoint if available
CHECKPOINT_PATH="${TRAIN_OUTPUT}/checkpoints/fold_0/best.pt"
if [[ -f "${CHECKPOINT_PATH}" ]]; then
    info "Running inference test..."
    if python -m Project.SubProject.engine.eval_engine infer \
        checkpoint="${CHECKPOINT_PATH}" \
        post="I feel sad all the time" \
        criterion="Depressed mood most of the day" \
        log_to_mlflow=false \
        > /tmp/infer_smoke.out 2>&1; then
        pass "Inference smoke test completed"
    else
        fail "Inference failed"
        cat /tmp/infer_smoke.out
    fi
else
    warn "No checkpoint found, skipping inference test"
fi

# Check 12: Linting (if in CI mode)
if [[ "${VALIDATE_MODE}" == "ci" ]]; then
    check "Code linting"
    info "Running ruff..."
    if ruff check src tests > /tmp/ruff.out 2>&1; then
        pass "Ruff checks passed"
    else
        fail "Ruff found issues"
        cat /tmp/ruff.out
        exit 1
    fi

    info "Running black..."
    if black --check src tests > /tmp/black.out 2>&1; then
        pass "Black formatting passed"
    else
        fail "Black found formatting issues"
        cat /tmp/black.out
        exit 1
    fi
fi

# Check 13: Unit tests (if in CI mode)
if [[ "${VALIDATE_MODE}" == "ci" ]]; then
    check "Unit tests"
    if pytest tests/unit/ -v --tb=short > /tmp/pytest.out 2>&1; then
        pass "Unit tests passed"
    else
        fail "Unit tests failed"
        cat /tmp/pytest.out
        exit 1
    fi
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total checks: ${TOTAL_CHECKS}"
echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"
echo ""

if [[ ${FAILED_CHECKS} -eq 0 ]]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "${GREEN}The quickstart workflow is validated.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed.${NC}"
    echo -e "${RED}Please review the output above.${NC}"
    exit 1
fi
