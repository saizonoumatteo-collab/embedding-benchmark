import sys, os, time, math, json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import httpx
import numpy as np

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    os.chdir(BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
OLLAMA_BASE_URL = "http://localhost:11434"

app = FastAPI(title="Embedding Benchmark")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ══════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════

STS_PAIRS = [
    ("A man is playing guitar.", "A person is playing a musical instrument.", 0.85),
    ("The stock market crashed today.", "Financial markets experienced a sharp decline.", 0.90),
    ("The recipe requires two eggs.", "You need a couple of eggs for this dish.", 0.88),
    ("The car engine stopped working.", "The vehicle broke down on the highway.", 0.82),
    ("The book was very interesting.", "I found the novel captivating.", 0.85),
    ("The meeting was postponed.", "The conference was rescheduled.", 0.88),
    ("Water boils at 100 degrees Celsius.", "H2O reaches its boiling point at 373 K.", 0.90),
    ("Children are playing in the park.", "Kids are having fun outdoors.", 0.87),
    ("The airplane landed safely.", "The flight arrived without incident.", 0.88),
    ("She is studying French.", "She is learning the French language.", 0.92),
    ("The singer performed at the concert.", "The musician entertained the audience live.", 0.88),
    ("The teacher explained the concept.", "The professor illustrated the idea clearly.", 0.82),
    ("The politician gave a speech.", "The official addressed the crowd.", 0.70),
    ("The dog ran across the field.", "The animal sprinted through the meadow.", 0.72),
    ("The project deadline is tomorrow.", "We need to submit the work by end of week.", 0.50),
    ("The hospital was built in 1952.", "Medical facilities were constructed decades ago.", 0.65),
    ("Fresh air is good for your health.", "Breathing outdoor air improves wellness.", 0.75),
    ("It is raining heavily outside.", "There is a light drizzle.", 0.50),
    ("He wrote a letter to his friend.", "She sent an email to her colleague.", 0.55),
    ("I bought a new laptop.", "He purchased a desktop computer.", 0.58),
    ("The athlete broke the world record.", "A competitor achieved a new personal best.", 0.65),
    ("The scientist discovered a new element.", "A researcher found an unknown chemical compound.", 0.78),
    ("She opened the window.", "He closed the door.", 0.28),
    ("I love pizza.", "Pasta is my favourite food.", 0.40),
    ("The sun sets in the west.", "The moon rises at night.", 0.18),
    ("The baby is sleeping.", "The infant is awake and crying.", 0.10),
    ("A cat is sitting on a mat.", "A dog is running in a park.", 0.10),
    ("She is learning to paint.", "He is studying music.", 0.22),
    ("He failed the exam.", "She passed with distinction.", 0.05),
    ("The mountain is covered in snow.", "The beach is warm and sunny.", 0.08),
]

RETRIEVAL_DATASET = [
    {"query": "How does machine learning work?",
     "documents": [
         "Machine learning algorithms learn statistical patterns from large training datasets.",
         "Neural networks are trained by back-propagating gradients to minimise a loss function.",
         "Supervised learning requires labelled examples to train a predictive model.",
         "Gradient descent iteratively adjusts model parameters to minimise prediction error.",
         "The Eiffel Tower stands 330 metres tall and was completed in 1889.",
         "Spaghetti carbonara is made with eggs, Pecorino cheese, guanciale and black pepper.",
         "The Amazon river discharges more water than any other river on Earth.",
         "Basketball was invented by Dr. James Naismith in Springfield in 1891.",
         "Mozart composed over 600 works including 41 symphonies.",
         "The average adult heart beats roughly 100 000 times per day.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "What are the effects of climate change?",
     "documents": [
         "Rising global temperatures are causing glaciers and polar ice caps to melt rapidly.",
         "Climate change intensifies extreme weather events such as hurricanes and droughts.",
         "Ocean acidification caused by CO2 absorption threatens marine ecosystems worldwide.",
         "Sea level rise due to thermal expansion endangers coastal communities globally.",
         "Photosynthesis is the process by which plants convert sunlight into glucose.",
         "The Great Wall of China stretches over 21 000 kilometres across northern China.",
         "The human genome contains approximately 3 billion base pairs of DNA.",
         "Baroque music flourished in Europe between approximately 1600 and 1750.",
         "Supply and demand determines the equilibrium price in a free market.",
         "Renaissance art emphasised realism, perspective and human anatomy.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "Tips for learning Python programming",
     "documents": [
         "Practice coding every day by building small projects to reinforce Python concepts.",
         "Use list comprehensions and generators to write more idiomatic Python code.",
         "Virtual environments isolate project dependencies to prevent version conflicts.",
         "Python decorators allow you to modify function behaviour without changing its source.",
         "The French Revolution began in 1789 with the storming of the Bastille.",
         "Yoga improves flexibility, strength and mental well-being through mindful movement.",
         "The Pacific Ocean covers more area than all the Earth's landmasses combined.",
         "Fermentation converts sugars into alcohol and CO2 using yeast or bacteria.",
         "Impressionist painters used short brushstrokes to capture light and movement.",
         "Compound interest grows exponentially over time, rewarding early savers.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "Nutrition advice for a healthy diet",
     "documents": [
         "Eating a variety of colourful vegetables ensures a wide range of micronutrients.",
         "Limiting processed foods and added sugars reduces the risk of chronic diseases.",
         "Adequate protein intake supports muscle repair and immune system function.",
         "Staying hydrated by drinking enough water is essential for metabolic processes.",
         "Quantum entanglement correlates particles regardless of the distance between them.",
         "The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512.",
         "Continental drift describes the slow movement of tectonic plates over millions of years.",
         "Jazz music originated in New Orleans in the early twentieth century.",
         "The Roman Empire at its peak controlled territories across three continents.",
         "A black hole's gravity is so strong that not even light can escape it.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "How to improve mental health and reduce stress",
     "documents": [
         "Regular physical exercise releases endorphins that naturally improve mood and reduce anxiety.",
         "Mindfulness meditation helps regulate the stress response by focusing on the present moment.",
         "Quality sleep of 7 to 9 hours per night is critical for emotional regulation.",
         "Building strong social connections provides a buffer against depression and isolation.",
         "Aerodynamic lift is generated by the pressure difference between wing surfaces.",
         "The Internet was developed from ARPANET, a US military network from the 1960s.",
         "Plate tectonics explains earthquakes, volcanic activity and mountain formation.",
         "The Silk Road connected China and Europe through a network of trade routes.",
         "Photovoltaic cells convert sunlight directly into electricity using semiconductors.",
         "Sourdough bread uses wild yeast and lactic acid bacteria for leavening.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "Latest advances in space exploration",
     "documents": [
         "Reusable rocket technology by SpaceX has dramatically reduced the cost of reaching orbit.",
         "The James Webb Space Telescope observes the universe in infrared from Lagrange point L2.",
         "NASA's Artemis programme aims to return humans to the Moon and establish a lunar gateway.",
         "Rovers like Perseverance search for biosignatures and cache samples for future return.",
         "The French language has approximately 300 million speakers across five continents.",
         "Ancient Egyptians built the pyramids as tombs for pharaohs over 4 500 years ago.",
         "Chess is believed to have originated in India around the 6th century AD.",
         "Insulin was discovered by Banting and Best in 1921 at the University of Toronto.",
         "The stock market crash of 1929 triggered the Great Depression across the world.",
         "Volcanoes form where tectonic plates diverge or where mantle plumes rise.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "Techniques for professional photography",
     "documents": [
         "The exposure triangle balances ISO, aperture and shutter speed to achieve correct exposure.",
         "The rule of thirds places subjects off-centre to create more dynamic compositions.",
         "Golden hour light just after sunrise or before sunset adds warm, flattering tones.",
         "RAW format retains all sensor data, allowing more flexibility in post-processing.",
         "Antibiotics kill or inhibit bacteria by targeting cell walls or protein synthesis.",
         "The speed of light in a vacuum is approximately 299 792 kilometres per second.",
         "The Great Barrier Reef is the world's largest coral reef system, spanning 2 300 km.",
         "Beethoven composed his Ninth Symphony after becoming completely deaf.",
         "GDP measures the total monetary value of goods and services produced in a country.",
         "Plate armour became the dominant form of knight protection in the 14th century.",
     ], "relevant_ids": [0, 1, 2, 3]},
    {"query": "Understanding blockchain and cryptocurrencies",
     "documents": [
         "A blockchain is a distributed ledger where transactions are recorded in immutable blocks.",
         "Proof of work requires miners to solve computational puzzles to validate new blocks.",
         "Smart contracts are self-executing programs stored on a blockchain without intermediaries.",
         "Cryptographic hashing ensures the integrity and immutability of blockchain data.",
         "The Battle of Waterloo in 1815 marked the final defeat of Napoleon Bonaparte.",
         "Cholesterol is a lipid molecule essential for cell membrane structure and hormone synthesis.",
         "The Sahara is the world's largest hot desert, covering most of northern Africa.",
         "Ballet originated in the Italian Renaissance courts of the fifteenth century.",
         "The periodic table organises elements by atomic number and chemical properties.",
         "The human eye can distinguish approximately 10 million different colours.",
     ], "relevant_ids": [0, 1, 2, 3]},
]

PARAPHRASE_GROUPS = [
    ["The customer service was absolutely outstanding.", "I was thoroughly impressed by the level of service I received.", "The support team provided an exceptional experience.", "I couldn't be happier with how I was treated as a customer."],
    ["This terrible weather is making me miserable.", "The awful conditions outside are ruining my day.", "I hate this dreadful, gloomy weather we're having.", "The horrendous climate today is utterly depressing."],
    ["I can't wait to travel and explore new places.", "The excitement of visiting foreign countries is unmatched.", "Travelling to new destinations fills me with joy.", "There is nothing I love more than discovering unknown places."],
    ["My computer keeps crashing and it is incredibly frustrating.", "I'm at my wit's end with this constant software failing.", "The endless technical glitches are driving me absolutely mad.", "My device is completely unreliable and I'm exhausted from fixing it."],
    ["Exercising regularly and eating well transformed my health.", "A balanced diet combined with daily workouts changed my life.", "I feel so much better since adopting a healthy lifestyle.", "Staying active and eating nutritious food improved my well-being enormously."],
    ["Getting promoted felt like all my hard work finally paid off.", "Years of dedication in my career led to a well-deserved advancement.", "The promotion was recognition of my consistent professional effort.", "Climbing the career ladder after persistent work was deeply satisfying."],
    ["We need to protect the environment for future generations.", "Preserving our planet is an urgent responsibility we all share.", "Sustainable practices are essential to safeguard the Earth's ecosystems.", "Taking action against pollution and deforestation is vital for tomorrow."],
    ["This meal is absolutely delicious and full of flavour.", "The food here tastes incredible, I am thoroughly enjoying it.", "Every bite of this dish is a wonderful culinary experience.", "The flavours in this recipe are perfectly balanced and deeply satisfying."],
    ["Continuous learning and reading expands the mind significantly.", "Studying new subjects keeps the brain sharp and adaptable.", "Acquiring knowledge through education opens countless doors in life.", "Dedicating time to learning enriches both personal and professional growth."],
    ["The new smartphone has a stunning display and impressive battery life.", "This phone's screen is gorgeous and it lasts all day on a single charge.", "The device features an excellent OLED panel and long-lasting power.", "With its vibrant screen and extended battery, this phone is outstanding."],
]

MULTILINGUAL_PAIRS = [
    ("The sun rises in the east every morning.", "Le soleil se lève à l'est chaque matin.", "EN→FR"),
    ("Knowledge is the most powerful tool we have.", "El conocimiento es la herramienta más poderosa que tenemos.", "EN→ES"),
    ("Water is essential for all forms of life.", "Wasser ist für alle Lebensformen unentbehrlich.", "EN→DE"),
    ("Love is the universal language of humanity.", "L'amore è il linguaggio universale dell'umanità.", "EN→IT"),
    ("Climate change threatens ecosystems worldwide.", "As alterações climáticas ameaçam os ecossistemas de todo o mundo.", "EN→PT"),
    ("Education is the foundation of a prosperous society.", "L'éducation est le fondement d'une société prospère.", "EN→FR"),
    ("Regular exercise improves physical and mental health.", "El ejercicio regular mejora la salud física y mental.", "EN→ES"),
    ("Technology is transforming every aspect of our lives.", "Die Technologie verändert jeden Aspekt unseres Lebens.", "EN→DE"),
    ("Teamwork and collaboration lead to better results.", "Il lavoro di squadra e la collaborazione portano a risultati migliori.", "EN→IT"),
    ("Protecting biodiversity is crucial for our future.", "Proteger a biodiversidade é fundamental para o nosso futuro.", "EN→PT"),
    ("Music has the power to heal and unite people.", "La musique a le pouvoir de guérir et d'unir les gens.", "EN→FR"),
    ("Artificial intelligence is revolutionising medicine.", "La inteligencia artificial está revolucionando la medicina.", "EN→ES"),
    ("Democracy requires an informed and engaged citizenry.", "Die Demokratie erfordert eine informierte und engagierte Bürgerschaft.", "EN→DE"),
    ("The ocean covers more than 70 percent of Earth's surface.", "L'oceano copre più del 70 percento della superficie terrestre.", "EN→IT"),
    ("Investing early leads to significant long-term wealth.", "Investir cedo leva a uma riqueza significativa a longo prazo.", "EN→PT"),
]

CLASSIFICATION_TEXTS = [
    ("The quarterback threw a game-winning touchdown in the final seconds.", "sports"),
    ("The tennis player won her third consecutive Grand Slam title.", "sports"),
    ("The basketball team mounted a stunning comeback in overtime.", "sports"),
    ("She completed the marathon in under three hours setting a new record.", "sports"),
    ("The cycling team dominated every mountain stage of the tour.", "sports"),
    ("The goalkeeper made an incredible save in the penalty shootout.", "sports"),
    ("A yellow card was shown to the midfielder for a reckless tackle.", "sports"),
    ("The swimmer broke the world record by a tenth of a second.", "sports"),
    ("The boxing match ended early when the champion landed a knockout punch.", "sports"),
    ("The gymnast scored a perfect ten on the balance beam routine.", "sports"),
    ("The new smartphone features a 200-megapixel sensor and satellite connectivity.", "technology"),
    ("Researchers developed a more efficient transformer architecture for language models.", "technology"),
    ("The critical firmware update patches several remote code execution vulnerabilities.", "technology"),
    ("Quantum computing reached a breakthrough milestone in logical qubit error correction.", "technology"),
    ("The solid-state battery electric vehicle can travel 800 km on a single charge.", "technology"),
    ("The open-source framework accelerates deployment of containerised microservices.", "technology"),
    ("Edge computing reduces latency by processing data closer to the source device.", "technology"),
    ("The augmented reality headset overlays digital information onto the physical world.", "technology"),
    ("Generative models can now produce photorealistic images from text descriptions.", "technology"),
    ("The chip manufacturer unveiled a processor with three nanometre transistors.", "technology"),
    ("Simmer the tomato sauce over low heat for at least twenty minutes.", "cooking"),
    ("Fold the egg whites gently to preserve air and achieve a light soufflé.", "cooking"),
    ("The sourdough starter needs to be fed daily before the long overnight proof.", "cooking"),
    ("Marinate the chicken thighs in lemon juice, garlic and fresh herbs.", "cooking"),
    ("Blend the ingredients until the batter is completely smooth and lump-free.", "cooking"),
    ("Deglaze the pan with white wine to lift the caramelised fond from the bottom.", "cooking"),
    ("Rest the steak for at least five minutes before slicing to retain its juices.", "cooking"),
    ("Roll the pasta dough until it is thin enough to see your hand through it.", "cooking"),
    ("Toast the spices in a dry pan to release their essential oils before grinding.", "cooking"),
    ("Caramelising onions slowly over low heat takes at least forty minutes.", "cooking"),
    ("The senator introduced a landmark healthcare reform bill in the upper chamber.", "politics"),
    ("Election results were contested following allegations of irregularities at polling stations.", "politics"),
    ("Parliament approved the national budget after a marathon overnight debate.", "politics"),
    ("The prime minister announced ambitious new net-zero carbon emission commitments.", "politics"),
    ("Diplomatic negotiations resumed between the two nations after a decade of hostility.", "politics"),
    ("The opposition party called for a vote of no confidence in the government.", "politics"),
    ("Trade tariffs on imported goods triggered a series of retaliatory measures.", "politics"),
    ("The constitutional court struck down the emergency decree as unconstitutional.", "politics"),
    ("A coalition government was formed following inconclusive general election results.", "politics"),
    ("The foreign minister condemned the military incursion as a violation of sovereignty.", "politics"),
    ("Scientists discovered a new species of deep-sea bioluminescent fish.", "science"),
    ("The space telescope captured the clearest images yet of a distant exoplanet atmosphere.", "science"),
    ("Clinical trials showed the mRNA vaccine reduced hospitalisation rates by 94 percent.", "science"),
    ("A new study links chronic sleep deprivation to accelerated cognitive decline.", "science"),
    ("Geologists uncovered evidence of an ancient super-volcano beneath the Antarctic ice sheet.", "science"),
    ("CRISPR gene editing successfully corrected a hereditary mutation in human embryo cells.", "science"),
    ("The particle accelerator detected a previously theoretical subatomic particle.", "science"),
    ("Astronomers confirmed the merger of two neutron stars using gravitational wave data.", "science"),
    ("A breakthrough in room-temperature superconductivity may transform energy transmission.", "science"),
    ("Researchers mapped the complete neural connectome of a mammalian brain region.", "science"),
]

SPEED_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming every major industry on the planet.",
    "Climate change poses an existential challenge to biodiversity and human civilisation.",
    "The neural network achieved state-of-the-art performance across multiple standard benchmarks.",
    "Quantum entanglement allows particles to be instantaneously correlated regardless of distance.",
    "The chef carefully plated the elaborate five-course tasting menu for the guests.",
    "International space exploration requires unprecedented scientific and logistical cooperation.",
    "Machine learning models discover complex patterns from large and varied datasets.",
    "Economic inequality has been rising consistently in many developed nations over recent decades.",
    "The concert hall resonated with the rich sound of the full symphony orchestra.",
    "Supply chain disruptions can propagate unpredictably across globalised manufacturing networks.",
    "Effective communication is the cornerstone of successful leadership and team cohesion.",
    "The discovery of penicillin by Alexander Fleming revolutionised the treatment of infections.",
    "Renewable energy sources such as solar and wind power are rapidly becoming cost-competitive.",
    "Understanding cognitive biases is essential for making better decisions under uncertainty.",
]

# ── NEW: Negation awareness pairs ─────────────────────────────────
NEGATION_PAIRS = [
    # Sentiment
    ("I absolutely love this restaurant, the food is fantastic.", "I absolutely hate this restaurant, the food is terrible.", "sentiment"),
    ("This product is amazing and I highly recommend it to everyone.", "This product is dreadful and I strongly advise everyone against it.", "sentiment"),
    ("The film was incredibly moving, beautiful and deeply inspiring.", "The film was utterly boring, ugly and profoundly disappointing.", "sentiment"),
    ("I feel fantastic, energetic and full of joy today.", "I feel terrible, exhausted and utterly miserable today.", "sentiment"),
    ("The concert was the best experience of my entire life.", "The concert was the worst experience of my entire life.", "sentiment"),
    # Factual opposites
    ("The economy is growing strongly with record employment levels.", "The economy is shrinking rapidly with record unemployment levels.", "factual"),
    ("The patient recovered fully and was discharged in excellent health.", "The patient deteriorated rapidly and passed away that evening.", "factual"),
    ("The team won the championship convincingly in front of a packed stadium.", "The team lost the championship embarrassingly in front of an empty stadium.", "factual"),
    ("The project was completed six months ahead of schedule and under budget.", "The project was abandoned six months before completion, massively over budget.", "factual"),
    ("Sales figures reached an all-time high, exceeding every forecast.", "Sales figures reached an all-time low, missing every forecast.", "factual"),
    # Logical negation
    ("All students passed the final examination with excellent grades.", "No students passed the final examination, all received failing grades.", "logical"),
    ("The bridge is open and traffic flows freely in both directions.", "The bridge is closed and traffic is completely blocked in both directions.", "logical"),
    ("She clearly remembered every detail of what happened that night.", "She had completely forgotten every detail of what happened that night.", "logical"),
    ("He arrived early and was fully prepared for the presentation.", "He arrived late and was completely unprepared for the presentation.", "logical"),
    ("The surgery was a complete success with no complications whatsoever.", "The surgery was a complete failure with severe complications throughout.", "logical"),
    # Moral judgment
    ("The politician was consistently honest, transparent and trustworthy.", "The politician was consistently dishonest, secretive and corrupt.", "moral"),
    ("He acted selflessly and generously to help those in need.", "He acted selfishly and greedily at the expense of those in need.", "moral"),
    ("The judge delivered a fair, impartial and just verdict.", "The judge delivered an unfair, biased and unjust verdict.", "moral"),
    ("She was proven completely innocent of all charges against her.", "She was proven completely guilty of all charges against her.", "moral"),
    ("The company treated its employees with dignity, respect and fairness.", "The company treated its employees with contempt, disrespect and cruelty.", "moral"),
]

# ── NEW: Topic drift ───────────────────────────────────────────────
TOPIC_DRIFT_SETS = [
    {
        "anchor": "Machine learning models learn patterns from large datasets to make predictions.",
        "levels": [
            "Deep learning uses neural networks with many layers to model complex patterns.",
            "Artificial intelligence encompasses both rule-based and data-driven approaches.",
            "Computer science covers algorithms, data structures and software engineering.",
            "Mathematics underpins computing through logic, statistics and linear algebra.",
            "Cooking a perfect risotto requires patience, technique and quality ingredients.",
        ],
        "expected": [0.85, 0.70, 0.50, 0.30, 0.05],
    },
    {
        "anchor": "Regular aerobic exercise significantly improves cardiovascular health and fitness.",
        "levels": [
            "Running, cycling and swimming are all excellent forms of aerobic conditioning.",
            "A healthy lifestyle combines physical activity with balanced nutrition and rest.",
            "Medicine and biology study the mechanisms of the human body and disease.",
            "Chemistry analyses the composition and properties of matter at a molecular level.",
            "The history of medieval Europe spans the fall of Rome to the Renaissance.",
        ],
        "expected": [0.82, 0.65, 0.45, 0.20, 0.05],
    },
    {
        "anchor": "Climate change is accelerating due to greenhouse gas emissions from human activity.",
        "levels": [
            "Global warming is causing more frequent and severe weather events worldwide.",
            "Environmental science studies the impact of human activity on natural ecosystems.",
            "Earth sciences include geology, oceanography, meteorology and atmospheric science.",
            "Physics explains the fundamental laws governing energy, matter and the universe.",
            "The latest blockbuster film broke box office records on its opening weekend.",
        ],
        "expected": [0.88, 0.68, 0.48, 0.25, 0.05],
    },
]

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def get_embedding(client: httpx.AsyncClient, model: str, text: str) -> list:
    try:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/embed",
                              json={"model": model, "input": text}, timeout=120)
        r.raise_for_status()
        d = r.json()
        if "embeddings" in d: return d["embeddings"][0]
        if "embedding"  in d: return d["embedding"]
    except Exception:
        pass
    r = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings",
                          json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def pearson_r(x, y) -> float:
    n = len(x)
    if n < 2: return 0.0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi-mx)**2 for xi in x) * sum((yi-my)**2 for yi in y))
    return num/den if den else 0.0


def ndcg_at_k(rel, k):
    dcg  = sum(r/math.log2(i+2) for i,r in enumerate(rel[:k]))
    idcg = sum(r/math.log2(i+2) for i,r in enumerate(sorted(rel,reverse=True)[:k]))
    return dcg/idcg if idcg else 0.0


def recall_at_k(rel, k):
    total = sum(rel)
    return sum(rel[:k])/total if total else 0.0


def mrr(rel):
    for i,r in enumerate(rel):
        if r: return 1.0/(i+1)
    return 0.0


def pca_2d(X: np.ndarray) -> list:
    X = X.astype(np.float64)
    X -= X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return (X @ Vt[:2].T).tolist()


def tsne_2d(X: np.ndarray, perplexity: float = 15.0, n_iter: int = 350) -> list:
    """Proper t-SNE in pure numpy."""
    n = X.shape[0]
    X = X.astype(np.float64)
    X -= X.mean(axis=0)
    std = X.std()
    if std > 0: X /= std

    sq = np.sum(X**2, axis=1)
    D2 = np.maximum(sq[:,None] + sq[None,:] - 2*(X @ X.T), 0.0)
    np.fill_diagonal(D2, 0.0)

    # Gaussian affinities with binary search for bandwidth
    P = np.zeros((n, n))
    log_perp = np.log(perplexity)
    for i in range(n):
        di = D2[i].copy(); di[i] = np.inf
        beta, lo, hi = 1.0, -np.inf, np.inf
        for _ in range(50):
            e = np.exp(-beta * di); e[i] = 0.0
            s = e.sum() + 1e-10
            pi = e / s
            mask = pi > 0
            H = -float(np.sum(pi[mask] * np.log(pi[mask] + 1e-10)))
            diff = H - log_perp
            if abs(diff) < 1e-5: break
            if diff > 0: lo=beta; beta=beta*2 if hi==np.inf else (beta+hi)/2
            else:        hi=beta; beta=beta/2 if lo==-np.inf else (beta+lo)/2
        P[i] = pi

    P = (P + P.T) / (2*n)
    P = np.maximum(P, 1e-12)
    P *= 4.0  # early exaggeration

    np.random.seed(42)
    Y = np.random.randn(n, 2) * 1e-4
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    lr = 200.0

    for t in range(n_iter):
        if t == 100: P /= 4.0

        sq_y = np.sum(Y**2, axis=1)
        D2_y = np.maximum(sq_y[:,None] + sq_y[None,:] - 2*(Y @ Y.T), 0.0)
        Q_num = 1.0 / (1.0 + D2_y)
        np.fill_diagonal(Q_num, 0.0)
        Q = Q_num / (Q_num.sum() + 1e-10)
        Q = np.maximum(Q, 1e-12)

        PQ   = (P - Q) * Q_num                          # (n,n)
        diff = Y[:,None,:] - Y[None,:,:]                # (n,n,2)
        grad = 4.0 * np.einsum('ij,ijk->ik', PQ, diff)  # (n,2)

        gains = (gains+0.2)*((grad>0)!=(dY>0)) + gains*0.8*((grad>0)==(dY>0))
        gains = np.maximum(gains, 0.01)
        dY    = 0.8*dY - lr*gains*grad
        Y    += dY
        Y    -= Y.mean(axis=0)

    return Y.tolist()


def silhouette_cosine(X: np.ndarray, labels: list) -> float:
    n = len(X)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn  = X / np.where(nrm==0, 1e-10, nrm)
    dist = 1.0 - Xn @ Xn.T
    la   = np.array(labels)
    scores = []
    for i in range(n):
        same = la == la[i]; same[i] = False
        if not same.any(): scores.append(0.0); continue
        a = dist[i][same].mean()
        b = min(dist[i][la==l].mean() for l in set(labels) if l != labels[i])
        mx = max(a, b)
        scores.append((b-a)/mx if mx else 0.0)
    return float(np.mean(scores))


def compute_overall_score(res: dict) -> Optional[float]:
    parts, weights = [], []
    if "speed"          in res and "latency_mean_ms"           in res["speed"]:
        parts.append(min(1.0, 1.0/(1.0+res["speed"]["latency_mean_ms"]/80)))
        weights.append(0.08)
    if "sts"            in res and "pearson_r"                  in res["sts"]:
        parts.append(max(0.0, res["sts"]["pearson_r"]))
        weights.append(0.18)
    if "classification" in res and "nearest_centroid_accuracy"  in res["classification"]:
        parts.append(res["classification"]["nearest_centroid_accuracy"])
        weights.append(0.12)
    if "retrieval"      in res and "ndcg_at_5"                  in res["retrieval"]:
        parts.append(res["retrieval"]["ndcg_at_5"])
        weights.append(0.28)
    if "robustness"     in res and "discrimination_ratio"        in res["robustness"]:
        parts.append(min(1.0, max(0.0, res["robustness"]["discrimination_ratio"]-1)))
        weights.append(0.09)
    if "multilingual"   in res and "alignment_score"            in res["multilingual"]:
        parts.append(min(1.0, max(0.0, res["multilingual"]["alignment_score"]+0.1)))
        weights.append(0.13)
    if "negation"       in res and "negation_awareness"          in res["negation"]:
        parts.append(res["negation"]["negation_awareness"])
        weights.append(0.07)
    if "topic_drift"    in res and "monotonicity_score"          in res["topic_drift"]:
        parts.append(res["topic_drift"]["monotonicity_score"])
        weights.append(0.05)
    if not parts: return None
    tw = sum(weights)
    return round(sum(p*w for p,w in zip(parts,weights))/tw*100, 1)


# ══════════════════════════════════════════════════════════════
# BENCHMARK RUNNERS
# ══════════════════════════════════════════════════════════════

async def run_speed(client, model, custom_texts=None):
    texts = custom_texts or SPEED_TEXTS
    latencies, dim = [], None
    for t in texts:
        t0 = time.perf_counter()
        emb = await get_embedding(client, model, t)
        latencies.append((time.perf_counter()-t0)*1000)
        if dim is None: dim = len(emb)
    lat = sorted(latencies)
    return {"latency_mean_ms": round(sum(latencies)/len(latencies),1),
            "latency_min_ms": round(lat[0],1), "latency_max_ms": round(lat[-1],1),
            "latency_p50_ms": round(lat[len(lat)//2],1),
            "latency_p95_ms": round(lat[int(len(lat)*0.95)],1),
            "throughput_per_sec": round(1000/(sum(latencies)/len(latencies)),2),
            "embedding_dim": dim, "num_texts": len(texts),
            "all_latencies": [round(l,1) for l in latencies]}


async def run_sts(client, model, custom_pairs=None):
    pairs = custom_pairs or STS_PAIRS
    pred, gt, details = [], [], []
    for ta, tb, score in pairs:
        ea = await get_embedding(client, model, ta)
        eb = await get_embedding(client, model, tb)
        sim = cosine_sim(ea, eb)
        pred.append(sim); gt.append(score)
        details.append({"text_a": ta, "text_b": tb, "predicted": round(sim,4), "ground_truth": score})
    pr = pearson_r(pred, gt)
    return {"pearson_r": round(pr,4), "num_pairs": len(pairs),
            "predicted": [round(v,4) for v in pred], "ground_truth": gt, "details": details}


async def run_retrieval(client, model):
    all_n5, all_n3, all_r5, all_m, per_query = [], [], [], [], []
    for item in RETRIEVAL_DATASET:
        qe  = await get_embedding(client, model, item["query"])
        des = [await get_embedding(client, model, d) for d in item["documents"]]
        sims = [cosine_sim(qe, de) for de in des]
        rel  = set(item["relevant_ids"])
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        rr = [1 if r in rel else 0 for r in ranked]
        n5 = ndcg_at_k(rr,5); n3 = ndcg_at_k(rr,3); r5 = recall_at_k(rr,5); m = mrr(rr)
        all_n5.append(n5); all_n3.append(n3); all_r5.append(r5); all_m.append(m)
        per_query.append({"query": item["query"], "ndcg@5": round(n5,4),
                          "ndcg@3": round(n3,4), "recall@5": round(r5,4), "mrr": round(m,4),
                          "top3": [item["documents"][ranked[i]] for i in range(min(3,len(ranked)))]})
    return {"ndcg_at_5": round(float(np.mean(all_n5)),4), "ndcg_at_3": round(float(np.mean(all_n3)),4),
            "recall_at_5": round(float(np.mean(all_r5)),4), "mrr": round(float(np.mean(all_m)),4),
            "num_queries": len(RETRIEVAL_DATASET), "per_query": per_query}


async def run_classification(client, model):
    texts   = [t for t,_ in CLASSIFICATION_TEXTS]
    lblstrs = [l for _,l in CLASSIFICATION_TEXTS]
    classes = sorted(set(lblstrs))
    l2i     = {l:i for i,l in enumerate(classes)}
    labels  = [l2i[l] for l in lblstrs]
    embs    = [await get_embedding(client, model, t) for t in texts]
    X       = np.array(embs)
    centroids = {l: X[[i for i,lb in enumerate(labels) if lb==l]].mean(axis=0) for l in set(labels)}

    correct = 0
    confusion = {c: {c2: 0 for c2 in classes} for c in classes}
    for i, e in enumerate(embs):
        pred_lbl = max(centroids, key=lambda l: cosine_sim(e, centroids[l]))
        pred_cls = classes[pred_lbl]; true_cls = lblstrs[i]
        confusion[true_cls][pred_cls] += 1
        if pred_lbl == labels[i]: correct += 1

    sil = silhouette_cosine(X, labels)
    pca = pca_2d(X)
    confusion_matrix = [[confusion[r][c] for c in classes] for r in classes]

    return {"nearest_centroid_accuracy": round(correct/len(embs),4),
            "silhouette_score": round(sil,4), "num_texts": len(texts),
            "num_classes": len(classes), "classes": classes,
            "pca_points": pca, "pca_labels": lblstrs,
            "confusion_matrix": confusion_matrix}


async def run_robustness(client, model):
    all_embs, glabels = [], []
    for gi, group in enumerate(PARAPHRASE_GROUPS):
        for t in group:
            all_embs.append(await get_embedding(client, model, t)); glabels.append(gi)
    X   = np.array(all_embs)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn  = X / np.where(nrm==0, 1e-10, nrm)
    sim_mat = Xn @ Xn.T
    n = len(all_embs)
    intra, inter = [], []
    for i in range(n):
        for j in range(i+1, n):
            (intra if glabels[i]==glabels[j] else inter).append(float(sim_mat[i,j]))
    avg_intra = float(np.mean(intra)) if intra else 0.0
    avg_inter = float(np.mean(inter)) if inter else 0.0
    ratio = avg_intra/avg_inter if avg_inter > 0 else 1.0
    per_group = []
    for gi, group in enumerate(PARAPHRASE_GROUPS):
        idxs = [i for i,l in enumerate(glabels) if l==gi]
        sims = [float(sim_mat[i,j]) for i in idxs for j in idxs if i<j]
        per_group.append({"texts": group, "avg_sim": round(float(np.mean(sims)),4) if sims else 0.0})
    return {"avg_intra_similarity": round(avg_intra,4), "avg_inter_similarity": round(avg_inter,4),
            "discrimination_ratio": round(ratio,4), "num_groups": len(PARAPHRASE_GROUPS),
            "texts_per_group": 4, "per_group": per_group}


async def run_multilingual(client, model):
    sims, details, en_embs, other_embs = [], [], [], []
    for en, other, pair in MULTILINGUAL_PAIRS:
        ea = await get_embedding(client, model, en)
        eb = await get_embedding(client, model, other)
        sim = cosine_sim(ea, eb)
        sims.append(sim); en_embs.append(ea); other_embs.append(eb)
        details.append({"en": en, "other": other, "lang": pair, "similarity": round(sim,4)})
    false_sims = [cosine_sim(en_embs[i], other_embs[(i+1)%len(MULTILINGUAL_PAIRS)]) for i in range(len(MULTILINGUAL_PAIRS))]
    avg = float(np.mean(sims)); avg_f = float(np.mean(false_sims))
    return {"avg_similarity": round(avg,4), "avg_non_translation_sim": round(avg_f,4),
            "alignment_score": round(avg-avg_f,4), "num_pairs": len(MULTILINGUAL_PAIRS), "details": details}


async def run_negation(client, model):
    by_cat: dict = {}
    all_sims = []
    details  = []
    for ta, tb, cat in NEGATION_PAIRS:
        ea = await get_embedding(client, model, ta)
        eb = await get_embedding(client, model, tb)
        sim = cosine_sim(ea, eb)
        all_sims.append(sim)
        by_cat.setdefault(cat, []).append(sim)
        details.append({"text_a": ta, "text_b": tb, "category": cat, "similarity": round(sim,4)})
    avg = float(np.mean(all_sims))
    cat_avg = {c: round(float(np.mean(v)),4) for c,v in by_cat.items()}
    # awareness = 1 - avg_negation_similarity (lower sim = more aware of negation)
    # We normalise: sim near 1 means model ignores negation, sim near 0 = model sees opposition
    awareness = round(1.0 - avg, 4)
    return {"avg_negation_similarity": round(avg,4), "negation_awareness": awareness,
            "num_pairs": len(NEGATION_PAIRS), "by_category": cat_avg,
            "details": details,
            "interpretation": "awareness > 0.6 = excellent · 0.4-0.6 = good · <0.4 = poor"}


async def run_topic_drift(client, model):
    results = []
    for ds in TOPIC_DRIFT_SETS:
        anchor_emb = await get_embedding(client, model, ds["anchor"])
        level_sims = []
        for lvl_text in ds["levels"]:
            lvl_emb = await get_embedding(client, model, lvl_text)
            level_sims.append(round(cosine_sim(anchor_emb, lvl_emb), 4))
        # Monotonicity score: fraction of consecutive pairs where sim decreases
        mono_count = sum(1 for i in range(len(level_sims)-1) if level_sims[i] > level_sims[i+1])
        mono_score = mono_count / (len(level_sims)-1)
        results.append({"anchor": ds["anchor"], "level_sims": level_sims,
                        "expected": ds["expected"], "mono_score": round(mono_score,4)})
    overall_mono = float(np.mean([r["mono_score"] for r in results]))
    # Correlation between actual and expected similarities
    actual_flat   = [s for r in results for s in r["level_sims"]]
    expected_flat = [s for r in results for s in r["expected"]]
    corr = pearson_r(actual_flat, expected_flat)
    return {"monotonicity_score": round(overall_mono,4), "correlation_with_expected": round(corr,4),
            "num_drift_sets": len(TOPIC_DRIFT_SETS), "per_set": results}


# ══════════════════════════════════════════════════════════════
# PYDANTIC
# ══════════════════════════════════════════════════════════════

class BenchmarkRequest(BaseModel):
    models:             list[str]
    run_speed:          bool = True
    run_sts:            bool = True
    run_retrieval:      bool = True
    run_classification: bool = True
    run_robustness:     bool = True
    run_multilingual:   bool = True
    run_negation:       bool = True
    run_topic_drift:    bool = True
    custom_texts:       Optional[list[str]] = None
    custom_sts_pairs:   Optional[list[list]] = None   # [[textA,textB,score],...]


class SimilarityRequest(BaseModel):
    model:  str; text_a: str; text_b: str


class VisualizeRequest(BaseModel):
    model:         str
    mode:          str = "pca"          # "pca" | "tsne" | "both"
    custom_texts:  Optional[list[str]] = None
    custom_labels: Optional[list[str]] = None


class HeatmapRequest(BaseModel):
    model:  str
    texts:  list[str]
    labels: Optional[list[str]] = None


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
async def health():
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return {"ollama": "ok", "status_code": r.status_code}
        except Exception as e:
            return {"ollama": "unreachable", "error": str(e)}


@app.get("/api/models")
async def list_models():
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10); r.raise_for_status()
        except Exception as e:
            raise HTTPException(502, str(e))
    all_m  = [m["name"] for m in r.json().get("models", [])]
    hints  = ["embed","mxbai","nomic","all-minilm","bge","gte","e5","snowflake","paraphrase","sentence","multilingual","jina"]
    tagged = [m for m in all_m if any(h in m.lower() for h in hints)]
    rest   = [m for m in all_m if m not in tagged]
    return {"models": tagged+rest, "embedding_tagged": tagged}


@app.post("/api/benchmark/stream")
async def stream_benchmark(req: BenchmarkRequest):
    custom_sts = [(p[0],p[1],p[2]) for p in req.custom_sts_pairs] if req.custom_sts_pairs else None

    async def generate():
        yield sse({"type": "start", "total_models": len(req.models)})
        async with httpx.AsyncClient() as client:
            for model in req.models:
                result: dict = {}
                yield sse({"type": "model_start", "model": model})

                TESTS = [
                    ("speed",          req.run_speed,          lambda: run_speed(client, model, req.custom_texts)),
                    ("sts",            req.run_sts,            lambda: run_sts(client, model, custom_sts)),
                    ("retrieval",      req.run_retrieval,      lambda: run_retrieval(client, model)),
                    ("classification", req.run_classification, lambda: run_classification(client, model)),
                    ("robustness",     req.run_robustness,     lambda: run_robustness(client, model)),
                    ("multilingual",   req.run_multilingual,   lambda: run_multilingual(client, model)),
                    ("negation",       req.run_negation,       lambda: run_negation(client, model)),
                    ("topic_drift",    req.run_topic_drift,    lambda: run_topic_drift(client, model)),
                ]
                for name, enabled, runner in TESTS:
                    if not enabled: continue
                    yield sse({"type": "test_start", "model": model, "test": name})
                    try:
                        data = await runner()
                        result[name] = data
                        yield sse({"type": "test_done", "model": model, "test": name, "data": data})
                    except Exception as e:
                        result[name] = {"error": str(e)}
                        yield sse({"type": "test_error", "model": model, "test": name, "error": str(e)})

                result["overall_score"] = compute_overall_score(result)
                yield sse({"type": "model_done", "model": model, "result": result})

        yield sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.post("/api/explore/similarity")
async def explore_similarity(req: SimilarityRequest):
    if not req.text_a.strip() or not req.text_b.strip():
        raise HTTPException(400, "Both texts required.")
    async with httpx.AsyncClient() as c:
        try:
            ea = await get_embedding(c, req.model, req.text_a)
            eb = await get_embedding(c, req.model, req.text_b)
        except Exception as e:
            raise HTTPException(502, str(e))
    sim = cosine_sim(ea, eb)
    return {"similarity": round(sim,6), "magnitude_a": round(float(np.linalg.norm(ea)),4),
            "magnitude_b": round(float(np.linalg.norm(eb)),4), "dim": len(ea)}


@app.post("/api/visualize")
async def visualize_embeddings(req: VisualizeRequest):
    if req.custom_texts:
        texts  = req.custom_texts
        labels = req.custom_labels or ["custom"]*len(texts)
    else:
        texts  = [t for t,_ in CLASSIFICATION_TEXTS]
        labels = [l for _,l in CLASSIFICATION_TEXTS]
    async with httpx.AsyncClient() as c:
        try: embs = [await get_embedding(c, req.model, t) for t in texts]
        except Exception as e: raise HTTPException(502, str(e))
    X = np.array(embs)
    out = {"labels": labels, "texts": texts,
           "pca_points": pca_2d(X)}
    if req.mode in ("tsne", "both"):
        out["tsne_points"] = tsne_2d(X)
    return out


@app.post("/api/heatmap")
async def compute_heatmap(req: HeatmapRequest):
    if len(req.texts) < 2:
        raise HTTPException(400, "At least 2 texts required.")
    if len(req.texts) > 60:
        raise HTTPException(400, "Maximum 60 texts for heatmap.")
    async with httpx.AsyncClient() as c:
        try: embs = [await get_embedding(c, req.model, t) for t in req.texts]
        except Exception as e: raise HTTPException(502, str(e))
    X   = np.array(embs, dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn  = X / np.where(nrm==0, 1e-10, nrm)
    mat = (Xn @ Xn.T).tolist()
    return {"matrix": mat, "texts": req.texts, "labels": req.labels or req.texts}
