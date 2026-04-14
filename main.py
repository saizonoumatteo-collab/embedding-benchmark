import asyncio
import sys
import os
import time
import json
import math
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import numpy as np

# Resolve base directory whether running normally or as a PyInstaller bundle
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
OLLAMA_BASE_URL = "http://localhost:11434"

app = FastAPI(title="Embedding Benchmark")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Datasets ──────────────────────────────────────────────────────────────────

STS_PAIRS = [
    ("A man is playing guitar.", "A person is playing a musical instrument.", 0.85),
    ("A cat is sitting on a mat.", "A dog is running in a park.", 0.10),
    ("The stock market crashed today.", "Financial markets experienced a sharp decline.", 0.90),
    ("She opened the window.", "He closed the door.", 0.30),
    ("The recipe requires two eggs.", "You need a couple of eggs for this dish.", 0.88),
    ("It is raining heavily outside.", "There is a light drizzle.", 0.55),
    ("The scientist discovered a new element.", "A researcher found an unknown chemical compound.", 0.80),
    ("I love pizza.", "Pasta is my favourite food.", 0.40),
    ("The car engine stopped working.", "The vehicle broke down on the highway.", 0.82),
    ("Children are playing in the park.", "Kids are having fun outdoors.", 0.87),
    ("The sun sets in the west.", "The moon rises at night.", 0.20),
    ("He wrote a letter to his friend.", "She sent an email to her colleague.", 0.60),
    ("The book was very interesting.", "I found the novel captivating.", 0.85),
    ("The airplane landed safely.", "The flight arrived without incident.", 0.88),
    ("Water boils at 100 degrees Celsius.", "H2O reaches its boiling point at 373 K.", 0.90),
    ("The athlete broke the world record.", "A competitor achieved a new personal best.", 0.65),
    ("The baby is sleeping.", "The infant is awake and crying.", 0.10),
    ("She is learning to paint.", "He is studying music.", 0.25),
    ("The meeting was postponed.", "The conference was rescheduled.", 0.88),
    ("I bought a new laptop.", "He purchased a desktop computer.", 0.60),
]

CLASSIFICATION_TEXTS = [
    # Sports
    ("The quarterback threw a touchdown pass in the final seconds.", "sports"),
    ("The tennis player won the Grand Slam tournament.", "sports"),
    ("The basketball team scored 120 points in overtime.", "sports"),
    ("She completed the marathon in under three hours.", "sports"),
    ("The cycling team dominated the mountain stage.", "sports"),
    # Technology
    ("The new smartphone features a 200-megapixel camera.", "technology"),
    ("Researchers developed a more efficient neural network architecture.", "technology"),
    ("The software update fixes critical security vulnerabilities.", "technology"),
    ("Quantum computing reached a new milestone in error correction.", "technology"),
    ("The electric vehicle can travel 500 miles on a single charge.", "technology"),
    # Cooking
    ("Simmer the sauce over low heat for 20 minutes.", "cooking"),
    ("Add a pinch of salt and fold the egg whites gently.", "cooking"),
    ("The sourdough bread needs to proof overnight.", "cooking"),
    ("Marinate the chicken in lemon juice and herbs.", "cooking"),
    ("Blend the ingredients until the texture is smooth.", "cooking"),
    # Politics
    ("The senator proposed a new healthcare reform bill.", "politics"),
    ("The election results were contested by opposition parties.", "politics"),
    ("Parliament approved the budget after a lengthy debate.", "politics"),
    ("The prime minister announced new climate commitments.", "politics"),
    ("Diplomatic talks resumed between the two nations.", "politics"),
    # Science
    ("Scientists discovered a new species of deep-sea fish.", "science"),
    ("The telescope captured images of a distant galaxy.", "science"),
    ("Clinical trials showed the vaccine is 95% effective.", "science"),
    ("The study links sleep deprivation to cognitive decline.", "science"),
    ("Geologists found evidence of ancient volcanic activity.", "science"),
]

SPEED_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming every industry.",
    "Climate change poses an existential threat to biodiversity.",
    "The neural network achieved state-of-the-art performance on several benchmarks.",
    "Quantum entanglement allows particles to be correlated regardless of distance.",
    "The chef carefully prepared the elaborate five-course meal.",
    "Space exploration requires unprecedented international cooperation.",
    "Machine learning models learn patterns from large datasets.",
    "Economic inequality has been rising steadily over the past decades.",
    "The concert hall was filled with the sound of a symphony orchestra.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

async def get_embedding(client: httpx.AsyncClient, model: str, text: str) -> list[float]:
    """Call Ollama /api/embed endpoint (v0.3+) with fallback to /api/embeddings."""
    payload = {"model": model, "input": text}
    try:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/embed", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # /api/embed returns {"embeddings": [[...]]}
        if "embeddings" in data:
            return data["embeddings"][0]
        if "embedding" in data:
            return data["embedding"]
    except Exception:
        pass

    # Legacy fallback
    payload_legacy = {"model": model, "prompt": text}
    r = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload_legacy, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _silhouette_cosine(X: np.ndarray, labels: list[int]) -> float:
    """Cosine-distance silhouette score implemented with pure numpy."""
    n = len(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    Xn = X / norms
    # cosine distance = 1 - cosine similarity
    dist = 1.0 - Xn @ Xn.T  # (n, n)
    labels_arr = np.array(labels)
    scores = []
    for i in range(n):
        same = labels_arr == labels_arr[i]
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = dist[i][same].mean()
        unique_other = [l for l in set(labels) if l != labels[i]]
        b = min(dist[i][labels_arr == l].mean() for l in unique_other)
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def pearson_r(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    return num / denom if denom else 0.0


# ── API Models ────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    models: list[str]
    run_speed: bool = True
    run_sts: bool = True
    run_classification: bool = True
    custom_texts: Optional[list[str]] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/models")
async def list_models():
    """Return only models that support embeddings (detected by tagging or trial)."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(502, f"Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")

    all_models = [m["name"] for m in r.json().get("models", [])]

    # Heuristic: known embedding model name fragments
    embedding_hints = [
        "embed", "mxbai", "nomic", "all-minilm", "bge", "gte", "e5",
        "snowflake-arctic", "paraphrase", "sentence", "multilingual",
    ]

    tagged, rest = [], []
    for name in all_models:
        lower = name.lower()
        if any(h in lower for h in embedding_hints):
            tagged.append(name)
        else:
            rest.append(name)

    return {"models": tagged + rest, "embedding_tagged": tagged}


@app.post("/api/benchmark")
async def run_benchmark(req: BenchmarkRequest):
    if not req.models:
        raise HTTPException(400, "Select at least one model.")

    results = {}

    async with httpx.AsyncClient() as client:
        for model in req.models:
            model_result: dict = {"model": model, "error": None}

            # ── Speed benchmark ────────────────────────────────────────────
            if req.run_speed:
                texts = req.custom_texts if req.custom_texts else SPEED_TEXTS
                latencies = []
                embedding_dim = None
                speed_error = None
                try:
                    for text in texts:
                        t0 = time.perf_counter()
                        emb = await get_embedding(client, model, text)
                        latencies.append((time.perf_counter() - t0) * 1000)
                        if embedding_dim is None:
                            embedding_dim = len(emb)
                    model_result["speed"] = {
                        "latency_mean_ms": round(sum(latencies) / len(latencies), 1),
                        "latency_min_ms": round(min(latencies), 1),
                        "latency_max_ms": round(max(latencies), 1),
                        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
                        "throughput_texts_per_sec": round(1000 / (sum(latencies) / len(latencies)), 2),
                        "embedding_dim": embedding_dim,
                        "num_texts": len(texts),
                    }
                except Exception as e:
                    model_result["speed"] = {"error": str(e)}

            # ── STS benchmark ──────────────────────────────────────────────
            if req.run_sts:
                predicted, ground_truth = [], []
                sts_error = None
                try:
                    for text_a, text_b, score in STS_PAIRS:
                        emb_a = await get_embedding(client, model, text_a)
                        emb_b = await get_embedding(client, model, text_b)
                        predicted.append(cosine_sim(emb_a, emb_b))
                        ground_truth.append(score)
                    pr = pearson_r(predicted, ground_truth)
                    model_result["sts"] = {
                        "pearson_r": round(pr, 4),
                        "num_pairs": len(STS_PAIRS),
                        "predicted_similarities": [round(v, 4) for v in predicted],
                        "ground_truth_similarities": ground_truth,
                    }
                except Exception as e:
                    model_result["sts"] = {"error": str(e)}

            # ── Classification benchmark ───────────────────────────────────
            if req.run_classification:
                try:
                    texts_cls = [t for t, _ in CLASSIFICATION_TEXTS]
                    labels_str = [l for _, l in CLASSIFICATION_TEXTS]

                    # Pure-numpy label encoding
                    unique_classes = sorted(set(labels_str))
                    label_to_idx = {l: i for i, l in enumerate(unique_classes)}
                    labels = [label_to_idx[l] for l in labels_str]

                    embeddings = []
                    for text in texts_cls:
                        emb = await get_embedding(client, model, text)
                        embeddings.append(emb)

                    emb_matrix = np.array(embeddings)

                    # Cosine-distance silhouette score (pure numpy)
                    sil = _silhouette_cosine(emb_matrix, labels)

                    # Nearest-centroid accuracy
                    label_set = list(set(labels))
                    centroids = {}
                    for lbl in label_set:
                        idxs = [i for i, l in enumerate(labels) if l == lbl]
                        centroids[lbl] = emb_matrix[idxs].mean(axis=0)

                    correct = 0
                    for i, emb in enumerate(embeddings):
                        sims = {lbl: cosine_sim(emb, c) for lbl, c in centroids.items()}
                        pred = max(sims, key=sims.get)
                        if pred == labels[i]:
                            correct += 1
                    accuracy = correct / len(embeddings)

                    model_result["classification"] = {
                        "silhouette_score": round(float(sil), 4),
                        "nearest_centroid_accuracy": round(accuracy, 4),
                        "num_texts": len(texts_cls),
                        "num_classes": len(label_set),
                        "classes": unique_classes,
                    }
                except Exception as e:
                    model_result["classification"] = {"error": str(e)}

            results[model] = model_result

    return {"results": results}


@app.get("/api/health")
async def health():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return {"ollama": "ok", "status_code": r.status_code}
        except Exception as e:
            return {"ollama": "unreachable", "error": str(e)}
