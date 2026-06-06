@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo Memulai Pelatihan Model AI (Next-Gen Trading Bot)
echo ========================================================

REM Pastikan uv terinstal. Jika belum, akan otomatis download dan install module yang dibutuhkan
echo Memeriksa dependensi Python...
set "UV_CACHE_DIR=%~dp0_uvcache"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
set "PYTHONUNBUFFERED=1"
set "TF_CPP_MIN_LOG_LEVEL=1"
set "STRICT_ARG="
if /I "%~1"=="/strict" (
    set "STRICT_ARG=--strict-eurusd-baseline"
    shift
)
if "%~1"=="" (
    set PAIRS=USDJPY EURJPY
) else (
    set PAIRS=%*
)

set "UV_EXITCODE=0"
for %%S in (%PAIRS%) do (
    echo Menjalankan training untuk %%S pada H1. Ini bisa lama, terutama di CPU-only Windows.
    "%USERPROFILE%\.local\bin\uv.exe" run --with pandas --with scikit-learn --with tensorflow --with tf2onnx --with onnx --with skl2onnx Python_Scripts\train_model.py --symbol %%S --timeframe H1 %STRICT_ARG%
    if errorlevel 1 (
        set "UV_EXITCODE=1"
        goto :after_training
    )
)

:after_training

echo.
echo ========================================================
if "%UV_EXITCODE%"=="0" (
    echo Proses selesai! Model ONNX Anda seharusnya sudah siap.
)
echo ========================================================
echo Exit code: %UV_EXITCODE%
if not "%UV_EXITCODE%"=="0" (
    echo Training gagal. Lihat output error di atas.
)
pause
