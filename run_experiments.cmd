@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM TwistNet-2D: Spiral-Twisted Channel Interactions for Texture Recognition
REM Full Benchmark Experiments
REM ============================================================================
REM
REM Datasets (5):
REM   - DTD: 47 textures, 10-fold (official)
REM   - FMD: 10 materials, 5-fold
REM   - KTH-TIPS2: 11 materials, 5-fold
REM   - CUB-200: 200 birds, 5-fold
REM   - Flowers-102: 102 flowers, 5-fold
REM
REM Models (8):
REM   Classic:
REM   - resnet18 (CVPR 2016) - 11.2M
REM   - se_resnet18 (CVPR 2018) - 11.3M
REM   Modern (2022-2024):
REM   - convnext (CVPR 2022) - 10.2M
REM   - hornet (NeurIPS 2022) - 13.5M
REM   - focalnet (NeurIPS 2022) - 10.5M
REM   - van (CVMJ 2023) - 13.4M
REM   - moganet (ICLR 2024) - 10.8M
REM   Ours:
REM   - twistnet18 - 12.0M
REM
REM Total: 8 models x (30 + 15*4) = 720 runs
REM Estimated time: ~150 GPU hours (RTX 3090)
REM ============================================================================

echo ============================================================================
echo TwistNet-2D Full Benchmark
echo ============================================================================
echo.

REM Check environment
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch not found. Please activate your environment:
    echo   conda activate twistnet2d
    pause
    exit /b 1
)

set MODELS=resnet18,se_resnet18,convnext,hornet,focalnet,van,moganet,twistnet18
set SEEDS=42,43,44
set EPOCHS=200

REM ============================================================================
REM DTD (10-fold official - REQUIRED for fair comparison with literature)
REM ============================================================================
echo.
echo [1/5] DTD: 8 models x 10 folds x 3 seeds = 240 runs
python run_all.py ^
    --data_dir data\dtd ^
    --dataset dtd ^
    --models %MODELS% ^
    --folds 1-10 ^
    --seeds %SEEDS% ^
    --epochs %EPOCHS% ^
    --run_dir runs\dtd

REM ============================================================================
REM FMD (5-fold)
REM ============================================================================
echo.
echo [2/5] FMD: 8 models x 5 folds x 3 seeds = 120 runs
python run_all.py ^
    --data_dir data\fmd ^
    --dataset fmd ^
    --models %MODELS% ^
    --folds 1-5 ^
    --seeds %SEEDS% ^
    --epochs %EPOCHS% ^
    --run_dir runs\fmd

REM ============================================================================
REM KTH-TIPS2 (5-fold)
REM ============================================================================
echo.
echo [3/5] KTH-TIPS2: 8 models x 5 folds x 3 seeds = 120 runs
python run_all.py ^
    --data_dir data\kth_tips2 ^
    --dataset kth_tips2 ^
    --models %MODELS% ^
    --folds 1-5 ^
    --seeds %SEEDS% ^
    --epochs %EPOCHS% ^
    --run_dir runs\kth_tips2

REM ============================================================================
REM CUB-200 (5-fold)
REM ============================================================================
echo.
echo [4/5] CUB-200: 8 models x 5 folds x 3 seeds = 120 runs
python run_all.py ^
    --data_dir data\cub200 ^
    --dataset cub200 ^
    --models %MODELS% ^
    --folds 1-5 ^
    --seeds %SEEDS% ^
    --epochs %EPOCHS% ^
    --run_dir runs\cub200

REM ============================================================================
REM Flowers-102 (5-fold)
REM ============================================================================
echo.
echo [5/5] Flowers-102: 8 models x 5 folds x 3 seeds = 120 runs
python run_all.py ^
    --data_dir data\flowers102 ^
    --dataset flowers102 ^
    --models %MODELS% ^
    --folds 1-5 ^
    --seeds %SEEDS% ^
    --epochs %EPOCHS% ^
    --run_dir runs\flowers102

REM ============================================================================
REM Summarize Results
REM ============================================================================
echo.
echo ============================================================================
echo Generating Summary Tables
echo ============================================================================

python summarize_runs.py --run_dir runs\dtd --dataset dtd --latex
python summarize_runs.py --run_dir runs\fmd --dataset fmd --latex
python summarize_runs.py --run_dir runs\kth_tips2 --dataset kth_tips2 --latex
python summarize_runs.py --run_dir runs\cub200 --dataset cub200 --latex
python summarize_runs.py --run_dir runs\flowers102 --dataset flowers102 --latex

echo.
echo ============================================================================
echo All experiments completed!
echo Results saved in: runs\
echo ============================================================================

endlocal
pause
