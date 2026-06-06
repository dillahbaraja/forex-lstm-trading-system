@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo Menjalankan Benchmark LSTM Terpisah
echo ========================================================

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
    echo Menjalankan benchmark untuk %%S pada H1. Ini bisa memakan waktu karena beberapa model dilatih ulang.
    "%USERPROFILE%\.local\bin\uv.exe" run --with pandas --with scikit-learn --with tensorflow --with tf2onnx --with onnx --with skl2onnx Python_Scripts\benchmark_lstm_models.py --symbol %%S --timeframe H1 %STRICT_ARG%
    if errorlevel 1 (
        set "UV_EXITCODE=1"
        goto :after_benchmark
    )
)

:after_benchmark

echo.
echo ========================================================
if "%UV_EXITCODE%"=="0" (
    echo Benchmark selesai. Laporan ada di folder benchmark_runs.
)
echo ========================================================
echo Exit code: %UV_EXITCODE%
if not "%UV_EXITCODE%"=="0" (
    echo Benchmark gagal. Lihat output error di atas.
)
pause
