@echo off
echo --- SUBIENDO CAMBIOS (GIT + DVC + JUPYTEXT) ---
echo.

REM 1. Preguntar mensaje de commit
set /p commit_msg="Escribe el mensaje del commit: "

if "%commit_msg%"=="" (
    echo [ERROR] El mensaje no puede estar vacio.
    pause
    exit
)

echo.
echo 1. Asegurando sincronizacion (ipynb -> py)...
REM Si se te olvido guardar el .py, esto lo hace por ti antes de subir
python -m jupytext --sync "*.ipynb" "*.py"

echo.
echo 2. Procesando datos con DVC...
dvc add DatasetsTFM/
dvc push

echo.
echo 3. Subiendo codigo a GitHub...
git add .
git commit -m "%commit_msg%"
git push

echo.
echo --- TODO SUBIDO CORRECTAMENTE ---
pause