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
    # ── 1. STANDARD baseline ────────────────────────────────────────
    {
        "query": "How does machine learning work?",
        "query_type": "standard",
        "documents": [
            "Machine learning algorithms learn statistical patterns from large training datasets.",        # 0 ✓
            "Neural networks are trained by back-propagating gradients to minimise a loss function.",     # 1 ✓
            "Supervised learning requires labelled examples to train a predictive model.",                # 2 ✓
            "Gradient descent iteratively adjusts model parameters to minimise prediction error.",        # 3 ✓
            "The Eiffel Tower stands 330 metres tall and was completed in 1889.",                         # 4 ✗ easy
            "Spaghetti carbonara is made with eggs, Pecorino cheese, guanciale and black pepper.",       # 5 ✗ easy
            "The Amazon river discharges more water than any other river on Earth.",                      # 6 ✗ easy
            "Basketball was invented by Dr. James Naismith in Springfield in 1891.",                     # 7 ✗ easy
            "Mozart composed over 600 works including 41 symphonies.",                                   # 8 ✗ easy
            "The average adult heart beats roughly 100 000 times per day.",                              # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2, 3],
        "hard_negative_ids": [],
    },
    # ── 2. HARD NEGATIVES — type 2 diabetes (related docs that are wrong) ──
    {
        "query": "What causes type 2 diabetes?",
        "query_type": "hard_negatives",
        "documents": [
            "Type 2 diabetes develops when cells become resistant to insulin and the pancreas cannot produce enough to compensate.",  # 0 ✓
            "Excess body weight, physical inactivity and a poor diet are the primary modifiable risk factors for type 2 diabetes.",  # 1 ✓
            "Chronically elevated blood glucose damages blood vessels and nerves, leading to the long-term complications of type 2 diabetes.",  # 2 ✓
            "Type 1 diabetes is an autoimmune condition where the immune system destroys insulin-producing beta cells in the pancreas.",  # 3 ✗ hard — diabetes but different type
            "Insulin therapy is essential for all patients diagnosed with type 1 diabetes from the point of diagnosis.",               # 4 ✗ hard — insulin/diabetes but for type 1
            "Gestational diabetes occurs during pregnancy due to hormonal changes and usually resolves after delivery.",               # 5 ✗ hard — diabetes subtype, not type 2
            "The pancreas secretes both digestive enzymes and the hormones insulin and glucagon into the bloodstream.",               # 6 ✗ hard — mentions pancreas/insulin, wrong context
            "A balanced diet rich in vegetables, whole grains and lean protein supports long-term weight management.",                # 7 ✗ semi-hard — health/diet but not diabetes
            "The stock market closed at a record high following positive corporate earnings reports.",                                 # 8 ✗ easy
            "Ancient Roman architecture continued to influence building design for many centuries.",                                  # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5, 6],
    },
    # ── 3. CONTRADICTION — exercise and mental health ───────────────
    {
        "query": "Does regular exercise improve mental health?",
        "query_type": "contradiction",
        "documents": [
            "Regular physical exercise releases endorphins and serotonin, significantly reducing symptoms of anxiety and depression.",   # 0 ✓
            "Studies show that 30 minutes of moderate exercise three times a week reduces depression risk by up to 47 percent.",        # 1 ✓
            "Exercise improves sleep quality and cognitive function, both of which are strongly linked to mental well-being.",          # 2 ✓
            "Exercise has no proven effect on mental health and should not be recommended as a treatment for depression.",              # 3 ✗ contradiction
            "Physical exertion increases cortisol levels and worsens anxiety disorders according to recent meta-analyses.",            # 4 ✗ contradiction
            "Antidepressant medication combined with cognitive behavioural therapy is effective for clinical depression.",              # 5 ✗ hard neg — mental health but different treatment
            "Mindfulness meditation reduces stress hormones and is proven to alleviate symptoms of anxiety.",                           # 6 ✗ hard neg — mental health but not exercise
            "The Great Wall of China stretches over 21 000 kilometres across northern China.",                                         # 7 ✗ easy
            "Supply chain logistics optimises the flow of goods from manufacturer to consumer.",                                       # 8 ✗ easy
            "Jazz music originated in New Orleans in the early twentieth century.",                                                    # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5, 6],
        "contradiction_ids": [3, 4],
    },
    # ── 4. MULTI-HOP — sleep and cognitive decline ──────────────────
    {
        "query": "What is the relationship between sleep deprivation and cognitive decline?",
        "query_type": "multi_hop",
        "documents": [
            "Chronic sleep deprivation impairs the brain's glymphatic system, reducing clearance of amyloid beta plaques overnight.",   # 0 ✓ hop-1
            "Accumulation of amyloid beta plaques in neural tissue is the primary pathological hallmark of Alzheimer's disease.",       # 1 ✓ hop-2 (combine with 0 to answer the query)
            "Adults who consistently sleep fewer than six hours per night show measurably accelerated cognitive decline over a decade.",# 2 ✓ direct answer
            "Sleep apnea causes repeated breathing interruptions during sleep and leads to chronic daytime fatigue.",                   # 3 ✗ hard — sleep disorder but not cognitive decline link
            "The brain consolidates procedural and declarative memories during deep and REM sleep stages.",                             # 4 ✗ hard — sleep + brain but about memory, not decline
            "Dementia affects approximately 55 million people worldwide and is the leading cause of disability in older age.",         # 5 ✗ hard — cognitive decline statistics but no sleep link
            "The speed of light in vacuum is approximately 299 792 kilometres per second.",                                            # 6 ✗ easy
            "Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight.",                               # 7 ✗ easy
            "The French Revolution began in 1789 with the storming of the Bastille fortress in Paris.",                               # 8 ✗ easy
            "Basketball was invented by Dr. James Naismith in Springfield, Massachusetts in 1891.",                                   # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5],
    },
    # ── 5. AMBIGUOUS QUERY — "Python performance" (language vs. snake) ──
    {
        "query": "Python performance and speed optimisation",
        "query_type": "ambiguous",
        "documents": [
            "Using NumPy vectorised operations instead of Python loops can accelerate numerical code by over 100 times.",               # 0 ✓ programming Python
            "Caching repeated function calls with functools.lru_cache dramatically reduces Python execution time.",                     # 1 ✓ programming Python
            "Profiling with cProfile and line_profiler reveals the exact bottlenecks in Python code for targeted optimisation.",       # 2 ✓ programming Python
            "Ball pythons are popular pets renowned for their docile temperament and manageable adult size.",                          # 3 ✗ distractor — snake, not programming
            "The reticulated python is the world's longest snake, capable of reaching lengths exceeding six metres.",                  # 4 ✗ distractor — snake "speed/size" context
            "Large constrictor snakes like pythons and anacondas can subdue prey far larger than themselves.",                         # 5 ✗ distractor — snake performance
            "Compiled languages like C++ and Rust are significantly faster than interpreted languages for CPU-intensive work.",        # 6 ✗ hard neg — performance but not Python
            "Database query optimisation with proper indexing reduces response times from seconds to milliseconds.",                   # 7 ✗ hard neg — optimisation but not Python
            "The Renaissance was a cultural movement spanning the 14th to 17th centuries in Europe.",                                  # 8 ✗ easy
            "Pasta carbonara is a traditional Roman dish made with guanciale, eggs and Pecorino Romano.",                             # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5, 6, 7],
        "distractor_ids": [3, 4, 5],
    },
    # ── 6. NOISY QUERY — green tea benefits with typos ──────────────
    {
        "query": "wat r the benifits of drnking grean tea evry day",
        "query_type": "noisy_query",
        "documents": [
            "Green tea contains catechin antioxidants that protect cells from oxidative damage and reduce inflammation.",               # 0 ✓
            "Daily green tea consumption is associated with reduced risk of cardiovascular disease and stroke.",                        # 1 ✓
            "The L-theanine in green tea promotes calm alertness and reduces stress without causing drowsiness.",                      # 2 ✓
            "Green tea polyphenols support metabolism and may assist with gradual weight management over time.",                       # 3 ✓
            "Black tea and green tea come from the same Camellia sinensis plant but differ in oxidation levels.",                     # 4 ✗ hard neg — tea but not green tea benefits
            "Caffeine sensitivity varies widely and can cause insomnia or anxiety in susceptible individuals.",                        # 5 ✗ hard neg — caffeine/tea adjacent but not benefits
            "Herbal teas such as chamomile and peppermint are caffeine-free alternatives to traditional teas.",                       # 6 ✗ hard neg — tea but not green tea
            "The Olympic Games originated in ancient Greece and were revived in Athens in 1896.",                                     # 7 ✗ easy
            "Quantum computers exploit superposition and entanglement to solve problems exponentially faster.",                        # 8 ✗ easy
            "Gothic architecture is characterised by pointed arches, ribbed vaults and flying buttresses.",                           # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2, 3],
        "hard_negative_ids": [4, 5, 6],
    },
    # ── 7. LONG DOCUMENT — vaccines and immune system ───────────────
    {
        "query": "How do vaccines stimulate the immune system?",
        "query_type": "long_document",
        "documents": [
            # Long relevant doc — the key challenge: can the model still rank this highly?
            "Vaccines work by introducing the immune system to a harmless form of a pathogen — this may be a weakened or "
            "inactivated virus, a recombinant protein subunit, or in the case of mRNA vaccines, genetic instructions that "
            "cause host cells to temporarily produce a recognisable antigen. Upon encountering this antigen, the innate immune "
            "system triggers a localised inflammatory response while the adaptive immune system activates B lymphocytes to "
            "produce pathogen-specific antibodies and cytotoxic T cells to eliminate infected cells. Crucially, a subset of "
            "these B and T cells differentiate into long-lived memory cells that persist for years or decades, allowing the "
            "immune system to mount a far more rapid and overwhelming secondary response upon any future exposure to the same "
            "real pathogen — this immunological memory is the fundamental mechanism that makes vaccination effective.",           # 0 ✓ long
            "Adjuvants are substances added to vaccines that enhance the immune response and ensure longer-lasting protection.", # 1 ✓ short
            "mRNA vaccines deliver synthetic genetic instructions that direct host cells to produce the target antigen.",        # 2 ✓ short
            "Antibiotics treat bacterial infections by inhibiting cell wall synthesis or disrupting protein production.",       # 3 ✗ hard — medical but antibiotics, not vaccines
            "Autoimmune diseases arise when the immune system mistakenly attacks the body's own healthy tissues.",              # 4 ✗ hard — immune system but wrong direction
            "Herd immunity is reached when a large enough proportion of a population has become immune to an infectious disease.",# 5 ✗ hard — vaccine-adjacent but population level
            "Inflammation is a protective immune response marked by redness, heat, swelling and pain at the injury site.",    # 6 ✗ hard — immune process but not vaccination mechanism
            "The Amazon rainforest produces approximately 20 percent of Earth's oxygen through photosynthesis.",               # 7 ✗ easy
            "Chess is a two-player strategy game that originated in northern India around the 6th century AD.",                # 8 ✗ easy
            "The printing press was invented by Johannes Gutenberg in Mainz around 1440.",                                     # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5, 6],
    },
    # ── 8. PARTIAL / INCOMPLETE QUERY — renewable energy storage ────
    {
        "query": "renewable energy storage",
        "query_type": "partial_query",
        "documents": [
            "Grid-scale lithium-ion battery systems store surplus solar and wind energy for dispatch during peak demand.",      # 0 ✓
            "Pumped hydroelectric storage accounts for over 90 percent of all installed grid energy storage worldwide.",       # 1 ✓
            "Green hydrogen produced via electrolysis using excess renewable energy can be stored and later converted to electricity.", # 2 ✓
            "Flow batteries offer scalable, long-duration energy storage well suited to integrating variable renewables.",    # 3 ✓
            "Solar panels convert sunlight into electricity using photovoltaic cells made from semiconductor silicon.",       # 4 ✗ hard neg — renewable but generation, not storage
            "Offshore wind farms generate electricity from strong, consistent winds above coastal waters.",                   # 5 ✗ hard neg — renewable but generation
            "The electrical grid continuously balances supply and demand across vast regional transmission networks.",        # 6 ✗ hard neg — grid but not storage
            "The Mona Lisa was painted by Leonardo da Vinci in the early sixteenth century.",                                  # 7 ✗ easy
            "Neurons communicate via electrical signals and chemical neurotransmitters across synaptic junctions.",          # 8 ✗ easy
            "The Second World War ended in 1945 with the unconditional surrender of Germany and Japan.",                     # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2, 3],
        "hard_negative_ids": [4, 5, 6],
    },
    # ── 9. SEMANTIC DISTRACTOR — interest rates (wrong sense of "interest") ──
    {
        "query": "How do interest rates affect the economy?",
        "query_type": "semantic_distractor",
        "documents": [
            "Higher interest rates increase borrowing costs, slowing consumer spending and business investment.",              # 0 ✓
            "Central banks raise interest rates to reduce inflation by cooling aggregate demand across the economy.",         # 1 ✓
            "Low interest rates stimulate economic growth by making credit cheaper for households and firms.",               # 2 ✓
            "Public interest in climate issues has grown substantially over the past decade.",                                # 3 ✗ distractor — "interest" but public opinion
            "The crime rate in major cities has been declining steadily since the 1990s.",                                    # 4 ✗ distractor — "rate" but crime rate
            "Compound interest causes savings to grow exponentially, rewarding long-term patient investors.",                # 5 ✗ hard neg — interest rate concept but personal finance
            "Bond prices and yields move inversely and are directly tied to central bank policy rate decisions.",            # 6 ✗ hard neg — financial, closely adjacent
            "The immune system defends the body against pathogens using specialised white blood cells.",                     # 7 ✗ easy
            "Volcanic eruptions release lava, pyroclastic ash and gases from the Earth's mantle.",                           # 8 ✗ easy
            "Ballet is a classical dance form that originated in the Italian Renaissance courts.",                           # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5, 6],
        "distractor_ids": [3, 4],
    },
    # ── 10. CONTRADICTION — human activity and climate change ───────
    {
        "query": "Is human activity responsible for climate change?",
        "query_type": "contradiction",
        "documents": [
            "97 percent of climate scientists agree that human greenhouse gas emissions are the primary driver of observed warming.",  # 0 ✓
            "The burning of fossil fuels since the Industrial Revolution has increased atmospheric CO2 concentrations by over 50 percent.", # 1 ✓
            "IPCC reports conclude with very high confidence that human activity is the dominant cause of warming since 1950.",       # 2 ✓
            "Climate change is a natural cycle entirely unrelated to human activity and driven by solar variability alone.",          # 3 ✗ contradiction
            "CO2 is a trace atmospheric gas with negligible effect on global temperatures; its role is greatly exaggerated.",        # 4 ✗ contradiction
            "Global average temperatures have risen by approximately 1.1 degrees Celsius since the pre-industrial era.",             # 5 ✗ hard neg — climate fact but not about causation
            "Renewable energy investment is growing rapidly worldwide to reduce dependence on fossil fuels.",                        # 6 ✗ hard neg — climate-adjacent, not about responsibility
            "The Silk Road was an ancient trade network linking China with Central Asia, the Middle East and Europe.",               # 7 ✗ easy
            "Baroque music is characterised by elaborate ornamentation, polyphony and the use of basso continuo.",                  # 8 ✗ easy
            "Pasta carbonara is a traditional Roman recipe using guanciale, eggs, Pecorino and black pepper.",                      # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5, 6],
        "contradiction_ids": [3, 4],
    },
    # ── 11. MULTI-HOP — antibiotic resistance ───────────────────────
    {
        "query": "How does antibiotic resistance develop in bacteria?",
        "query_type": "multi_hop",
        "documents": [
            "Bacteria reproduce rapidly and random mutations occasionally produce individuals with reduced sensitivity to an antibiotic.", # 0 ✓ hop-1
            "When antibiotics eliminate sensitive bacteria while resistant variants survive and reproduce, natural selection causes resistant strains to dominate.", # 1 ✓ hop-2
            "Overuse and incomplete courses of antibiotics accelerate the emergence and spread of resistant bacterial strains.",          # 2 ✓ direct
            "Antibiotics are ineffective against viral infections such as the common cold, influenza or COVID-19.",                      # 3 ✗ hard — antibiotic limitation but not resistance mechanism
            "Mutations in the BRCA1 gene significantly increase the lifetime risk of breast and ovarian cancer.",                        # 4 ✗ hard — mutation but cancer context
            "Vaccination programmes have successfully eradicated smallpox and nearly eliminated polio worldwide.",                       # 5 ✗ hard — infectious disease control but not resistance
            "The Louvre museum in Paris houses over 35 000 artworks, including the Mona Lisa.",                                          # 6 ✗ easy
            "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials.",                              # 7 ✗ easy
            "The Pacific Ocean is the largest ocean on Earth, covering approximately one-third of its surface.",                        # 8 ✗ easy
            "Impressionism was an art movement that emerged in France during the 1860s and 1870s.",                                     # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2],
        "hard_negative_ids": [3, 4, 5],
    },
    # ── 12. STANDARD — space exploration ────────────────────────────
    {
        "query": "Latest advances in space exploration",
        "query_type": "standard",
        "documents": [
            "Reusable rocket technology pioneered by SpaceX has dramatically reduced the cost of reaching orbit.",            # 0 ✓
            "The James Webb Space Telescope observes the universe in infrared wavelengths from Lagrange point L2.",           # 1 ✓
            "NASA's Artemis programme aims to return humans to the lunar surface and establish a permanent gateway.",         # 2 ✓
            "Rovers like Perseverance search for biosignatures and cache samples for eventual return to Earth.",              # 3 ✓
            "The French language is spoken by approximately 300 million people across five continents.",                     # 4 ✗ easy
            "Ancient Egyptians constructed the pyramids as monumental tombs for pharaohs over 4 500 years ago.",            # 5 ✗ easy
            "Chess is believed to have originated in northern India around the 6th century AD.",                             # 6 ✗ easy
            "Insulin was discovered by Banting and Best in 1921 at the University of Toronto.",                              # 7 ✗ easy
            "The stock market crash of 1929 was a major trigger of the global Great Depression.",                           # 8 ✗ easy
            "Volcanoes form at tectonic plate boundaries and above mantle hotspots.",                                        # 9 ✗ easy
        ],
        "relevant_ids": [0, 1, 2, 3],
        "hard_negative_ids": [],
    },
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
    if "retrieval" in res and "ndcg_at_5" in res["retrieval"]:
        r = res["retrieval"]
        # Composite retrieval score: standard NDCG weighted with harder signals
        components, cw = [r["ndcg_at_5"]], [0.5]
        if r.get("top1_accuracy")           is not None: components.append(r["top1_accuracy"]);           cw.append(0.2)
        if r.get("hard_negative_rejection") is not None: components.append(r["hard_negative_rejection"]); cw.append(0.15)
        if r.get("contradiction_rejection") is not None: components.append(r["contradiction_rejection"]); cw.append(0.1)
        if r.get("distractor_rejection")    is not None: components.append(r["distractor_rejection"]);    cw.append(0.05)
        retrieval_score = sum(c*w for c,w in zip(components,cw)) / sum(cw)
        parts.append(retrieval_score)
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
    all_n5, all_n3, all_r5, all_m = [], [], [], []
    top1_hits = []
    hard_hn_rejections: list[float] = []   # per-query hard-neg rejection fraction
    hard_n5: list[float] = []              # ndcg@5 only for queries with hard negs
    contradiction_rejections: list[float] = []
    distractor_rejections: list[float] = []
    per_query = []
    type_scores: dict[str, list] = {}

    for item in RETRIEVAL_DATASET:
        qe   = await get_embedding(client, model, item["query"])
        des  = [await get_embedding(client, model, d) for d in item["documents"]]
        sims = [cosine_sim(qe, de) for de in des]

        rel      = set(item["relevant_ids"])
        hn_ids   = set(item.get("hard_negative_ids", []))
        cont_ids = set(item.get("contradiction_ids", []))
        dist_ids = set(item.get("distractor_ids", []))

        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        rank_of = {doc_id: rank for rank, doc_id in enumerate(ranked)}

        rr = [1 if r in rel else 0 for r in ranked]
        n5 = ndcg_at_k(rr, 5)
        n3 = ndcg_at_k(rr, 3)
        r5 = recall_at_k(rr, 5)
        m  = mrr(rr)
        all_n5.append(n5); all_n3.append(n3); all_r5.append(r5); all_m.append(m)

        # Top-1 accuracy
        top1_hits.append(1 if ranked[0] in rel else 0)

        # Hard negative rejection: fraction of hard negs ranked below the worst-ranked relevant doc
        if hn_ids:
            worst_rel_rank = max(rank_of[r] for r in rel)
            hn_below = sum(1 for h in hn_ids if rank_of[h] > worst_rel_rank)
            hard_hn_rejections.append(hn_below / len(hn_ids))
            hard_n5.append(n5)

        # Contradiction rejection
        if cont_ids:
            worst_rel_rank = max(rank_of[r] for r in rel)
            cont_below = sum(1 for c in cont_ids if rank_of[c] > worst_rel_rank)
            contradiction_rejections.append(cont_below / len(cont_ids))

        # Distractor rejection (ambiguous / semantic distractor types)
        if dist_ids:
            worst_rel_rank = max(rank_of[r] for r in rel)
            dist_below = sum(1 for d in dist_ids if rank_of[d] > worst_rel_rank)
            distractor_rejections.append(dist_below / len(dist_ids))

        qtype = item.get("query_type", "standard")
        type_scores.setdefault(qtype, []).append(n5)

        per_query.append({
            "query":       item["query"],
            "query_type":  qtype,
            "ndcg@5":      round(n5, 4),
            "ndcg@3":      round(n3, 4),
            "recall@5":    round(r5, 4),
            "mrr":         round(m, 4),
            "top1":        ranked[0] in rel,
            "top3_docs":   [item["documents"][ranked[i]][:80] for i in range(min(3, len(ranked)))],
            "top3_ids":    ranked[:3],
            "top3_relevant": [ranked[i] in rel for i in range(min(3, len(ranked)))],
        })

    # Aggregate new metrics
    top1_acc = float(np.mean(top1_hits))
    hn_reject = float(np.mean(hard_hn_rejections)) if hard_hn_rejections else None
    ndcg_hard = float(np.mean(hard_n5))            if hard_n5            else None
    cont_rej  = float(np.mean(contradiction_rejections)) if contradiction_rejections else None
    dist_rej  = float(np.mean(distractor_rejections))    if distractor_rejections    else None
    type_ndcg = {t: round(float(np.mean(v)), 4) for t, v in type_scores.items()}

    out = {
        "ndcg_at_5":                round(float(np.mean(all_n5)), 4),
        "ndcg_at_3":                round(float(np.mean(all_n3)), 4),
        "recall_at_5":              round(float(np.mean(all_r5)), 4),
        "mrr":                      round(float(np.mean(all_m)),  4),
        "top1_accuracy":            round(top1_acc, 4),
        "ndcg_at_5_hard_only":      round(ndcg_hard, 4) if ndcg_hard is not None else None,
        "hard_negative_rejection":  round(hn_reject, 4) if hn_reject  is not None else None,
        "contradiction_rejection":  round(cont_rej,  4) if cont_rej   is not None else None,
        "distractor_rejection":     round(dist_rej,  4) if dist_rej   is not None else None,
        "ndcg_by_type":             type_ndcg,
        "num_queries":              len(RETRIEVAL_DATASET),
        "per_query":                per_query,
    }
    return out


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
