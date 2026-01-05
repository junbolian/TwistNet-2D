@echo off
REM ============================================================================
REM TwistNet-2D Quick Test
REM Validates setup before running full experiments
REM ============================================================================
REM
REM What it does:
REM   - Tests model building (all 8 models)
REM   - Runs short training on DTD fold 1 (ResNet + TwistNet)
REM   - Validates numerical stability
REM
REM Expected time: ~20 minutes on single GPU
REM ============================================================================

echo ============================================================================
echo TwistNet-2D Quick Validation Test
echo ============================================================================
echo.

REM Check environment
echo [Step 1/4] Checking environment...
python -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch not found!
    echo Please run: conda activate twistnet2d
    pause
    exit /b 1
)

REM Test model builds
echo.
echo [Step 2/4] Testing model builds...
python -c "from models import build_model, count_params; import torch; x = torch.randn(2,3,224,224); models=['resnet18','se_resnet18','convnext','hornet','focalnet','van','moganet','twistnet18']; [print(f'  {n}: {count_params(build_model(n,47))/1e6:.1f}M OK') for n in models]"
if errorlevel 1 (
    echo [ERROR] Model build test failed!
    pause
    exit /b 1
)

REM Test dataset loading
echo.
echo [Step 3/4] Testing dataset loading...
python -c "from datasets import get_dataloaders; from transforms import build_train_transform, build_eval_transform; t1=build_train_transform(); t2=build_eval_transform(); tr,va,te,nc = get_dataloaders('data/dtd','dtd',1,t1,t2); print(f'  DTD: train={len(tr.dataset)}, val={len(va.dataset)}, test={len(te.dataset)}, classes={nc}')" 2>nul
if errorlevel 1 (
    echo [WARNING] Dataset loading failed - make sure data/dtd exists
    echo Skipping dataset test...
) else (
    echo   Dataset loading OK
)

REM Run quick training
echo.
echo [Step 4/4] Running quick training test (50 epochs)...
echo   This will train ResNet-18 and TwistNet-18 on DTD fold 1
echo.

python run_all.py ^
    --data_dir data\dtd ^
    --dataset dtd ^
    --models resnet18,twistnet18 ^
    --folds 1 ^
    --seeds 42 ^
    --epochs 50 ^
    --run_dir runs_test

echo.
echo ============================================================================
echo Quick test completed!
echo ============================================================================
echo.
echo Check results:
echo   - runs_test\dtd_fold1_resnet18_seed42\results.json
echo   - runs_test\dtd_fold1_twistnet18_seed42\results.json
echo.
echo If both models trained successfully (no NaN/Inf), proceed with:
echo   run_experiments.cmd
echo ============================================================================

pause
