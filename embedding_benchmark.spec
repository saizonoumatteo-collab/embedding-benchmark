# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

hidden_imports = [
    # uvicorn
    "uvicorn", "uvicorn.main", "uvicorn.config", "uvicorn.server",
    "uvicorn.logging", "uvicorn.lifespan", "uvicorn.lifespan.off",
    "uvicorn.lifespan.on", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.loops.asyncio", "uvicorn.protocols",
    "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.middleware", "uvicorn.middleware.proxy_headers",
    # fastapi / starlette
    "fastapi", "starlette", "starlette.responses", "starlette.routing",
    "starlette.staticfiles", "starlette.middleware",
    "starlette.middleware.cors", "starlette.background",
    "starlette.concurrency", "starlette.exceptions",
    "starlette.datastructures", "starlette.types",
    "starlette.requests", "starlette.websockets",
    "starlette.status", "starlette.templating",
    # pydantic
    "pydantic", "pydantic.v1",
    # anyio
    "anyio", "anyio._backends._asyncio", "anyio._backends._trio",
    "anyio.abc", "anyio.streams", "anyio.streams.memory",
    # httpx
    "httpx", "httpx._transports", "httpx._transports.default",
    "httpx._transports.asgi", "httpx._transports.wsgi",
    # numpy
    "numpy", "numpy.core", "numpy.lib",
    # misc
    "h11", "multiprocessing", "multiprocessing.util",
    "email.mime.text", "email.mime.multipart",
    "asyncio", "logging", "logging.handlers",
]

datas = [
    (os.path.join("static"), "static"),
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "tensorflow", "torch", "pytest", "rich"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="EmbeddingBenchmark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
