@echo off
echo ===========================================
echo   Embedding Benchmark - Demarrage
echo ===========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    pause
    exit /b 1
)

:: Install dependencies if needed
if not exist ".venv" (
    echo [INFO] Creation de l'environnement virtuel...
    python -m venv .venv
)

echo [INFO] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat

echo [INFO] Installation des dependances...
pip install -r requirements.txt --quiet

echo.
echo [INFO] Demarrage du serveur sur http://localhost:8000
echo [INFO] Appuyez sur Ctrl+C pour arreter.
echo.

start "" http://localhost:8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
