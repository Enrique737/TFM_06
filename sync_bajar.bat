@echo off
echo --- ACTUALIZANDO PROYECTO (GIT + DVC + JUPYTEXT) ---
echo.

echo 1. Bajando cambios de codigo (Git)...
git pull
if %errorlevel% neq 0 goto :error

echo.
echo 2. Sincronizando Notebooks (py -> ipynb)...
REM Esto actualiza tus notebooks locales con el codigo nuevo que acaba de bajar
python -m jupytext --sync "**/*.ipynb" "**/*.py"
if %errorlevel% neq 0 goto :error

echo.
echo 3. Bajando datos pesados (DVC)...
dvc pull
if %errorlevel% neq 0 goto :error

echo.
echo --- PROCESO COMPLETADO EXITOSAMENTE ---
pause
exit

:error
echo.
echo [ERROR] Ha ocurrido un problema. Revisa los mensajes de arriba.
pause