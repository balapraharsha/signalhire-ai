"""
jd_config.py — All JD-derived constants for SignalHire AI.

Extracted from job_description.docx by careful reading.
These are NOT keyword lists — they encode the JD's INTENT.
"""

# ── HARD TECHNICAL REQUIREMENTS ─────────────────────────────────────────────
# Things the JD says "you absolutely need"
HARD_REQUIRED_SKILLS = [
    # Embedding / retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "embedding",
    "BGE", "E5", "OpenAI embeddings", "text embeddings",
    # Vector DBs / hybrid search
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "FAISS", "Elasticsearch",
    "OpenSearch", "vector database", "vector search", "hybrid search",
    # Ranking / retrieval systems
    "ranking", "retrieval", "search", "recommendation", "recommender",
    "NDCG", "MRR", "MAP", "evaluation", "A/B testing", "A/B test",
    # Python
    "Python",
    # ML core
    "machine learning", "ML", "NLP", "natural language processing",
    "LLM", "large language model", "RAG", "retrieval augmented",
    "transformer", "fine-tuning", "fine tuning", "LoRA", "PEFT",
]

# Semantic clusters for embedding-based matching
# These are the CONCEPTS the JD is looking for
JD_CONCEPT_SENTENCES = [
    "Production experience with embeddings-based retrieval systems deployed to real users.",
    "Production experience with vector databases or hybrid search infrastructure.",
    "Hands-on experience designing evaluation frameworks for ranking systems using NDCG MRR MAP.",
    "Strong Python programming skills and code quality.",
    "Applied machine learning engineer who ships production systems.",
    "Natural language processing and information retrieval experience.",
    "LLM fine-tuning experience with LoRA QLoRA PEFT.",
    "Learning to rank models XGBoost neural ranking.",
    "Experience with recommendation systems at scale.",
    "Candidate job description matching and ranking systems.",
    "AI engineer at a product company building intelligent systems.",
    "Semantic search and dense retrieval systems.",
]

# ── SOFT / CULTURE SIGNALS ───────────────────────────────────────────────────
# JD explicitly values these
PRODUCT_COMPANY_SIGNALS = [
    # Direct indicators
    "startup", "Series A", "Series B", "product company", "SaaS",
    # Known India product companies
    "Flipkart", "Meesho", "Swiggy", "Zomato", "Ola", "PhonePe", "Razorpay",
    "Cred", "Groww", "Zepto", "Blinkit", "Nykaa", "Myntra", "Lenskart",
    "Sarvam", "Krutrim", "Juspay", "Khatabook", "BrowserStack", "Postman",
    "Freshworks", "Zoho", "Chargebee", "Hasura", "Unacademy", "Byju",
    "upGrad", "Vedantu", "Clevertap", "MoEngage", "Sprinklr", "Darwinbox",
    "Leadsquared", "Zenoti", "Capillary", "InMobi", "Sharechat",
    # Global product companies
    "Google", "Meta", "Microsoft", "Amazon", "Apple", "Netflix", "Uber",
    "Airbnb", "Stripe", "Databricks", "Snowflake", "Confluent", "HashiCorp",
    "Notion", "Figma", "Linear", "Vercel", "Hugging Face",
]

# ── EXPLICIT DISQUALIFIERS ───────────────────────────────────────────────────
# JD says "we explicitly do NOT want" or "will not move forward"
CONSULTING_COMPANIES = [
    "TCS", "Tata Consultancy", "Infosys", "Wipro", "Accenture",
    "Cognizant", "Capgemini", "HCL", "Mphasis", "Hexaware",
    "Mindtree", "L&T Infotech", "LTIMindtree", "Tech Mahindra",
    "Persistent", "Mastech", "NIIT Technologies", "Zensar",
    "Cyient", "Mphasis", "KPIT", "Sasken",
]

DISQUALIFIER_TITLES = [
    "marketing manager", "marketing executive", "marketing analyst",
    "hr manager", "hr executive", "human resources",
    "accountant", "finance", "financial analyst",
    "content writer", "content creator", "content strategist",
    "graphic designer", "ui designer", "ux designer",
    "sales executive", "sales manager", "business development",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "customer support", "customer success",
    "operations manager", "supply chain",
    "project manager",  # soft disqualifier — not hard block but lower weight
]

# Skills that are red flags when they're the ONLY skills (wrong domain)
WRONG_DOMAIN_SKILLS = [
    "photoshop", "illustrator", "figma design", "sketch",
    "excel", "powerpoint", "word",
    "SEO", "social media marketing", "google ads", "facebook ads",
    "autocad", "solidworks", "matlab simulation",
    "SAP", "oracle ERP", "salesforce CRM",
    "cobol", "mainframe",
]

# Pure research disqualifiers — "will not move forward"
RESEARCH_ONLY_SIGNALS = [
    "PhD candidate", "research scientist", "research engineer",
    "postdoctoral", "postdoc", "academic research",
]

# ── JOB LOGISTICS ────────────────────────────────────────────────────────────
PREFERRED_LOCATIONS = [
    "Pune", "Noida", "Hyderabad", "Mumbai", "Delhi",
    "Delhi NCR", "Gurugram", "Gurgaon", "Bangalore", "Bengaluru",
    "Noida", "Greater Noida",
]

# JD says "sub-30-day preferred, can buy out up to 30 days"
# "30+ day notice candidates still in scope but bar gets higher"
IDEAL_NOTICE_DAYS = 30
MAX_NOTICE_DAYS = 90   # beyond this, logistics score drops sharply

# JD says 5-9 years, but "some people hit senior judgment at 4 years"
IDEAL_YOE_MIN = 5
IDEAL_YOE_MAX = 9
ABSOLUTE_MIN_YOE = 3   # below this, very unlikely

# Salary band — reasonable for Series A Senior AI Engineer in India
SALARY_BAND_MIN_LPA = 25
SALARY_BAND_MAX_LPA = 80

# ── SCORING WEIGHTS (log-damped JD emphasis) ─────────────────────────────────
# These come FROM the JD text frequency analysis, not manual tuning.
# The JD spends most text on: retrieval/ranking/embeddings (technical)
# then career type (product vs services)
# then behavioral (active, responsive)
# logistics are mentioned but briefly
FEATURE_BASE_WEIGHTS = {
    "semantic_match":      1.0,   # anchor — normalized relative to this
    "profile_coherence":   0.85,
    "career_consistency":  0.70,
    "expertise_depth":     0.80,
    "engagement_decay":    0.75,
    "behavioral_score":    0.65,
    "antipattern_penalty": 1.50,  # applied as multiplier, not additive
    "logistics_fit":       0.40,
}
