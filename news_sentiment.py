"""
news_sentiment.py — Fetch crypto news sentiment dari Tavily API.
"""

import json
import logging
import os
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SENTIMENT_PATH = DATA_DIR / "news_sentiment.json"
ENV_PATH = BASE_DIR / ".env"

# Tavily API Config
TAVILY_API_URL = "https://api.tavily.com/search"

# ── ROBUST ENV LOADER ─────────────────────────────────────────────

def load_env(path: Path = None) -> bool:
    """Load .env file dengan multiple fallback paths."""
    
    # Coba multiple lokasi
    possible_paths = [
        Path(__file__).parent / ".env",           # Same folder as script
        Path.cwd() / ".env",                       # Current working directory
        Path.home() / ".env",                      # Home directory
        ENV_PATH                                   # Default config
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            log.info(f"📄 Found .env at: {env_path}")
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and val:
                            os.environ.setdefault(key, val)
                            log.info(f"✅ Loaded: {key}={val[:20]}...")
                log.info("✅ .env loaded successfully")
                return True
            except Exception as e:
                log.warning(f"⚠️  Failed to load {env_path}: {e}")
    
    log.warning("❌ .env file not found in any location")
    return False

# Load .env immediately
load_env()

# Get API Key
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Search queries untuk crypto news
SEARCH_QUERIES = [
    "Bitcoin BTC price analysis news 24 hours",
    "cryptocurrency market sentiment today",
    "BTC USDT trading news latest",
    "crypto regulation news Bitcoin",
    "Bitcoin institutional investment news"
]

# ── TAVILY API CLIENT ─────────────────────────────────────────────

def fetch_tavily_news(query: str, days: int = 1) -> Optional[Dict]:
    """Fetch news dari Tavily API."""
    if not TAVILY_API_KEY:
        log.error("❌ TAVILY_API_KEY tidak ditemukan")
        return None
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 5,
        "days": days,
        "include_domains": [],
        "exclude_domains": []
    }
    
    try:
        response = requests.post(TAVILY_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        log.info("✅ Tavily: %d results for '%s'", len(data.get("results", [])), query[:50])
        return data
    
    except requests.exceptions.RequestException as e:
        log.error("❌ Tavily API error: %s", str(e))
        return None


def analyze_sentiment(results: List[Dict]) -> Dict:
    """Analisis sentimen dari news results."""
    if not results:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "total_articles": 0,
            "summary": "No news available"
        }
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    titles = []
    
    for result in results:
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        
        positive_keywords = [
            "bullish", "rise", "gain", "up", "increase", "positive", "growth",
            "rally", "surge", "jump", "soar", "breakthrough", "adoption",
            "approval", "partnership", "investment", "institutional"
        ]
        
        negative_keywords = [
            "bearish", "drop", "fall", "down", "decrease", "negative", "crash",
            "decline", "plunge", "sell-off", "regulation", "ban", "warning",
            "risk", "loss", "hack", "scam", "lawsuit"
        ]
        
        text = f"{title} {content}"
        
        pos_score = sum(1 for kw in positive_keywords if kw in text)
        neg_score = sum(1 for kw in negative_keywords if kw in text)
        
        if pos_score > neg_score:
            positive_count += 1
        elif neg_score > pos_score:
            negative_count += 1
        else:
            neutral_count += 1
        
        titles.append(result.get("title", ""))
    
    total = positive_count + negative_count + neutral_count
    
    if total > 0:
        sentiment_score = (positive_count - negative_count) / total
    else:
        sentiment_score = 0.0
    
    if sentiment_score > 0.2:
        sentiment_label = "BULLISH"
    elif sentiment_score < -0.2:
        sentiment_label = "BEARISH"
    else:
        sentiment_label = "NEUTRAL"
    
    return {
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_label": sentiment_label,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_articles": total,
        "top_titles": titles[:5],
        "summary": f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral articles",
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


def fetch_all_news_sentiment() -> Dict:
    """Fetch news dari semua queries dan aggregate sentiment."""
    log.info("📰 Fetching crypto news sentiment from Tavily...")
    
    all_results = []
    
    for query in SEARCH_QUERIES:
        data = fetch_tavily_news(query, days=1)
        if data and data.get("results"):
            all_results.extend(data["results"][:3])
    
    seen_titles = set()
    unique_results = []
    for result in all_results:
        title = result.get("title", "")
        if title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(result)
    
    log.info("📊 Total unique articles: %d", len(unique_results))
    
    sentiment = analyze_sentiment(unique_results)
    sentiment["articles"] = unique_results[:10]
    
    DATA_DIR.mkdir(exist_ok=True)
    with open(SENTIMENT_PATH, "w", encoding="utf-8") as f:
        json.dump(sentiment, f, indent=2, default=str)
    
    log.info("✅ Sentiment saved to: %s", SENTIMENT_PATH)
    log.info("📈 Sentiment: %s (Score: %.2f)", sentiment["sentiment_label"], sentiment["sentiment_score"])
    
    return sentiment


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TAVILY NEWS SENTIMENT TEST")
    print("=" * 60)
    
    # Debug: Show API Key status
    print(f"\n🔍 API Key Status:")
    print(f"   TAVILY_API_KEY in env: {'✅ YES' if TAVILY_API_KEY else '❌ NO'}")
    print(f"   API Key length: {len(TAVILY_API_KEY)} chars")
    print(f"   API Key prefix: {TAVILY_API_KEY[:15]}..." if TAVILY_API_KEY else "   N/A")
    print(f"   .env path: {ENV_PATH}")
    print(f"   .env exists: {ENV_PATH.exists()}")
    
    if not TAVILY_API_KEY:
        print("\n❌ ERROR: TAVILY_API_KEY tidak ditemukan!")
        print("\n Solusi:")
        print("   1. Pastikan .env ada di folder yang sama dengan news_sentiment.py")
        print("   2. Format: TAVILY_API_KEY=tvly-dev-xxx")
        print("   3. Tidak ada spasi sebelum/sesudah =")
        print("   4. Restart terminal setelah update .env")
        print("   5. Atau set manual: set TAVILY_API_KEY=tvly-dev-xxx")
        print("\n" + "=" * 60)
    else:
        print(f"\n✅ API Key detected: {TAVILY_API_KEY[:20]}...")
        print("\n📰 Fetching news from Tavily...")
        sentiment = fetch_all_news_sentiment()
        
        print("\n" + "=" * 60)
        print("  NEWS SENTIMENT ANALYSIS")
        print("=" * 60)
        print(f"  Sentiment     : {sentiment['sentiment_label']}")
        print(f"  Score         : {sentiment['sentiment_score']:.2f}")
        print(f"  Positive      : {sentiment['positive_count']}")
        print(f"  Negative      : {sentiment['negative_count']}")
        print(f"  Neutral       : {sentiment['neutral_count']}")
        print(f"  Total Articles: {sentiment['total_articles']}")
        print("=" * 60)
        print(f"\n✅ Sentiment saved to: {SENTIMENT_PATH}")
