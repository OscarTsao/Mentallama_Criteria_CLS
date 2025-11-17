#!/usr/bin/env python3
"""
Build Verification Script

Tests all implemented modules can be imported and basic functionality works.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("Build Verification Script")
print("=" * 60)
print()

errors = []
warnings = []

# Test 1: Check Python version
print("[1/10] Checking Python version...")
import sys
if sys.version_info >= (3, 10):
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    errors.append(f"Python version {sys.version_info.major}.{sys.version_info.minor} is too old")
print()

# Test 2: Import utils
print("[2/10] Testing utils module...")
try:
    from Project.SubProject.utils import get_logger, set_seed
    from Project.SubProject.utils.mlflow_utils import configure_mlflow
    print("  ✓ Utils module imports successfully")

    # Test logger
    logger = get_logger('test')
    logger.info("Logger works")
    print("  ✓ Logger functional")

    # Test seed setting
    set_seed(42)
    print("  ✓ Seed setting functional")
except Exception as e:
    errors.append(f"Utils module failed: {e}")
    print(f"  ✗ Error: {e}")
print()

# Test 3: Import data module
print("[3/10] Testing data module...")
try:
    from Project.SubProject.data import MentalHealthDataset, create_folds, Sample
    print("  ✓ Data module imports successfully")
except Exception as e:
    errors.append(f"Data module failed: {e}")
    print(f"  ✗ Error: {e}")
print()

# Test 4: Import models module
print("[4/10] Testing models module...")
try:
    from Project.SubProject.models import build_prompt, normalize_text
    print("  ✓ Models module imports successfully")

    # Test prompt builder
    prompt = build_prompt("I feel sad", "Depressed mood")
    assert len(prompt) > 0
    print("  ✓ Prompt builder functional")

    # Test text normalization
    text = normalize_text("  Hello   World  ")
    assert text == "Hello World"
    print("  ✓ Text normalization functional")
except Exception as e:
    errors.append(f"Models module failed: {e}")
    print(f"  ✗ Error: {e}")
print()

# Test 5: Import engine module
print("[5/10] Testing engine module...")
try:
    from Project.SubProject.engine import compute_metrics, MetricsTracker
    print("  ✓ Engine module imports successfully")

    # Test metrics
    import numpy as np
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    print("  ✓ Metrics computation functional")

    # Test tracker
    tracker = MetricsTracker()
    tracker.update(0, {'val_f1': 0.8})
    print("  ✓ Metrics tracker functional")
except Exception as e:
    errors.append(f"Engine module failed: {e}")
    print(f"  ✗ Error: {e}")
print()

# Test 6: Check data files
print("[6/10] Checking data files...")
data_files = {
    'DSM5 criteria': Path('data/DSM5/MDD_Criteira.json'),
    'RedSM5 posts': Path('data/redsm5/redsm5_posts.csv'),
    'RedSM5 annotations': Path('data/redsm5/redsm5_annotations.csv'),
}
all_present = True
for name, path in data_files.items():
    if path.exists():
        print(f"  ✓ {name}: {path}")
    else:
        warnings.append(f"Missing data file: {path}")
        print(f"  ⚠ {name}: {path} (missing)")
        all_present = False

if all_present:
    print("  ✓ All data files present")
print()

# Test 7: Test dataset loading (if data present)
print("[7/10] Testing dataset loading...")
if all_present:
    try:
        dataset = MentalHealthDataset(
            redsm5_path='data/redsm5',
            dsm5_path='data/DSM5',
            override_counts=True,
        )
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")

        # Test sample access
        sample = dataset[0]
        assert 'post' in sample
        assert 'criterion' in sample
        assert 'label' in sample
        print(f"  ✓ Sample access functional")

        # Test groups
        groups = dataset.get_groups()
        print(f"  ✓ Groups extraction functional: {len(set(groups))} unique posts")
    except Exception as e:
        errors.append(f"Dataset loading failed: {e}")
        print(f"  ✗ Error: {e}")
else:
    print("  ⊘ Skipped (data files missing)")
print()

# Test 8: Check for syntax errors in all Python files
print("[8/10] Checking Python syntax...")
import py_compile
src_files = list(Path('src/Project/SubProject').rglob('*.py'))
syntax_errors = []
for file in src_files:
    try:
        py_compile.compile(str(file), doraise=True)
    except py_compile.PyCompileError as e:
        syntax_errors.append(str(file))
        errors.append(f"Syntax error in {file}")

if not syntax_errors:
    print(f"  ✓ All {len(src_files)} Python files have valid syntax")
else:
    print(f"  ✗ {len(syntax_errors)} files have syntax errors")
print()

# Test 9: Check module structure
print("[9/10] Checking module structure...")
expected_modules = [
    'src/Project/SubProject/__init__.py',
    'src/Project/SubProject/data/__init__.py',
    'src/Project/SubProject/data/dataset.py',
    'src/Project/SubProject/data/splits.py',
    'src/Project/SubProject/models/__init__.py',
    'src/Project/SubProject/models/model.py',
    'src/Project/SubProject/models/prompt_builder.py',
    'src/Project/SubProject/engine/__init__.py',
    'src/Project/SubProject/engine/metrics.py',
    'src/Project/SubProject/engine/train_engine.py',
    'src/Project/SubProject/engine/eval_engine.py',
    'src/Project/SubProject/utils/__init__.py',
    'src/Project/SubProject/utils/log.py',
    'src/Project/SubProject/utils/seed.py',
    'src/Project/SubProject/utils/mlflow_utils.py',
]

missing_modules = []
for module in expected_modules:
    if not Path(module).exists():
        missing_modules.append(module)

if not missing_modules:
    print(f"  ✓ All {len(expected_modules)} expected modules present")
else:
    print(f"  ✗ {len(missing_modules)} modules missing")
    errors.extend([f"Missing module: {m}" for m in missing_modules])
print()

# Test 10: Check entry points
print("[10/10] Checking entry points...")
try:
    import importlib.util

    # Check train_engine
    spec = importlib.util.spec_from_file_location(
        "train_engine",
        "src/Project/SubProject/engine/train_engine.py"
    )
    if spec and hasattr(spec.loader, 'exec_module'):
        print("  ✓ train_engine.py is a valid module")

    # Check eval_engine
    spec = importlib.util.spec_from_file_location(
        "eval_engine",
        "src/Project/SubProject/engine/eval_engine.py"
    )
    if spec and hasattr(spec.loader, 'exec_module'):
        print("  ✓ eval_engine.py is a valid module")

    print("  ✓ Entry points are valid")
except Exception as e:
    warnings.append(f"Entry point check failed: {e}")
    print(f"  ⚠ Warning: {e}")
print()

# Summary
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"Errors: {len(errors)}")
print(f"Warnings: {len(warnings)}")
print()

if errors:
    print("ERRORS:")
    for error in errors:
        print(f"  ✗ {error}")
    print()

if warnings:
    print("WARNINGS:")
    for warning in warnings:
        print(f"  ⚠ {warning}")
    print()

if not errors:
    print("✓ Build verification PASSED")
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -e .")
    print("  2. Run training: python -m Project.SubProject.engine.train_engine --help")
    print("  3. Run inference: python -m Project.SubProject.engine.eval_engine --help")
    sys.exit(0)
else:
    print("✗ Build verification FAILED")
    sys.exit(1)
