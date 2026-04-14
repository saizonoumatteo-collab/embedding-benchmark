# ⚡ Embedding Benchmark

Plateforme de benchmark pour les modèles d'embedding disponibles via **[Ollama](https://ollama.com)**.  
Interface web moderne, résultats en temps réel, export JSON/CSV.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Fonctionnalités

| Benchmark | Métriques |
|-----------|-----------|
| 🚀 **Vitesse** | Latence moy. / min / P95 / max, throughput (textes/sec), dimension des vecteurs |
| 🎯 **Similarité sémantique (STS)** | Corrélation de Pearson entre les similarités cosinus prédites et des scores humains (20 paires annotées) |
| 📦 **Classification** | Score Silhouette (cosinus) + précision centroïde sur 25 textes répartis en 5 catégories |

- Détection automatique des modèles Ollama (badge `embed` pour les modèles d'embedding connus)
- Comparaison multi-modèles simultanée
- Graphiques interactifs (Chart.js) : barres, distribution de latence, scatter STS
- Export des résultats en **JSON** et **CSV**
- Textes personnalisés pour le benchmark de vitesse

---

## Prérequis

- [Ollama](https://ollama.com/download) installé et en cours d'exécution
- Au moins un modèle d'embedding pulled, par exemple :

```bash
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull all-minilm
```

---

## Démarrage rapide

### Option A — EXE Windows (aucune installation requise)

Télécharge `EmbeddingBenchmark.exe` depuis les [Releases](../../releases), double-clique dessus.  
Le serveur démarre et le navigateur s'ouvre automatiquement sur `http://localhost:8000`.

### Option B — Python (toutes plateformes)

```bash
# 1. Cloner le repo
git clone https://github.com/saizonoumatteo-collab/embedding-benchmark.git
cd embedding-benchmark

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Ou sous Windows, double-cliquer sur **`start.bat`** (crée un venv et installe automatiquement).

---

## Architecture

```
embedding-benchmark/
├── main.py                  # Backend FastAPI (API + logique de benchmark)
├── launcher.py              # Point d'entrée PyInstaller (ouvre le navigateur)
├── requirements.txt
├── start.bat                # Lanceur Windows
├── embedding_benchmark.spec # Spec PyInstaller pour générer l'EXE
└── static/
    ├── index.html
    ├── css/styles.css
    └── js/app.js
```

**Stack** : FastAPI · Uvicorn · httpx · NumPy · Chart.js

---

## Construire l'EXE

```bash
# Dans un environnement virtuel propre
pip install fastapi uvicorn[standard] httpx numpy python-multipart pyinstaller
pyinstaller embedding_benchmark.spec --clean
# → dist/EmbeddingBenchmark.exe
```

---

## API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web |
| `/api/models` | GET | Liste les modèles Ollama disponibles |
| `/api/benchmark` | POST | Lance un benchmark (body JSON) |
| `/api/health` | GET | Statut de connexion Ollama |

### Exemple de requête benchmark

```json
POST /api/benchmark
{
  "models": ["nomic-embed-text", "mxbai-embed-large"],
  "run_speed": true,
  "run_sts": true,
  "run_classification": true,
  "custom_texts": null
}
```

---

## Licence

MIT
