"""
Entry point for the PyInstaller bundle.
Starts the FastAPI server via uvicorn and opens the browser.
"""
import multiprocessing
multiprocessing.freeze_support()

import sys
import os
import time
import threading
import webbrowser
import uvicorn

# When frozen by PyInstaller, sys._MEIPASS is the temp extraction directory.
# We patch the working directory so that FastAPI can find static/.
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    os.chdir(BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HOST = "127.0.0.1"
PORT = 8000
URL  = f"http://{HOST}:{PORT}"


def _open_browser():
    time.sleep(1.8)
    webbrowser.open(URL)


if __name__ == "__main__":
    # Import the app object directly — uvicorn cannot resolve the "main:app"
    # string import inside a PyInstaller bundle.
    from main import app

    print(f"[Embedding Benchmark] Demarrage sur {URL} ...")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
    )
