import json, os, time, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd
import anthropic
import spacy

nlp = spacy.load("en_core_web_sm")
MODEL = "claude-sonnet-4-6-20250514"

def get_client():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

DROP_POS = {"DET", "AUX", "INTJ", "SCONJ", "CCONJ", "PART", "ADP"}
FILLER = {
    "just", "really", "basically", "actually", "simply", "essentially",
    "generally", "however", "furthermore", "additionally", "certainly",
    "obviously", "literally", "honestly", "frankly", "clearly",
    "sure", "happy", "please", "perhaps",
    "also", "very", "quite", "rather", "somewhat", "only",
    "new", "already", "even", "still", "now", "then",
    "many", "much", "more", "most", "well", "often",
    "since", "while", "during", "nearly", "several",
}
LIGHT_VERBS = {
    "has", "have", "had", "is", "was", "were", "are", "been",
    "do", "does", "did", "make", "made", "get", "got",
    "take", "took", "give", "gave", "say", "said", "says",
    "include", "includes", "included", "involve", "involves",
    "according", "added", "noted", "told", "called",
    "saw", "seen", "come", "came", "went", "go",
}
DROP_ADJ = {
    "other", "former", "first", "last", "second", "third",
    "same", "own", "next", "previous", "recent", "current",
    "entire", "total", "full", "whole", "major", "main",
    "particular", "specific", "certain", "various",
    "possible", "likely", "known", "related", "based",
    "annual", "daily", "early", "late", "long", "short",
}


def compress(text):
    doc = nlp(text)
    out = []
    for sent in doc.sents:
        tokens = []
        for t in sent:
            if t.is_punct or t.pos_ in DROP_POS:
                continue
            if t.lower_ in FILLER or t.lower_ in LIGHT_VERBS:
                continue
            if t.is_stop and t.pos_ in ("PRON", "ADV"):
                continue
            if t.pos_ == "ADJ" and t.lower_ in DROP_ADJ:
                continue
            tokens.append(t.text)
        if tokens:
            out.append(" ".join(tokens))
    return ". ".join(out) + "." if out else text


N = 150
BASE = Path(__file__).parent
DATA = BASE / "data"
OUT = BASE / "results"
CAVEMAN = "Respond in caveman-speak. Drop articles, filler, hedging. Fragments OK. Keep all facts."
CONDITIONS = ["baseline", "output_compressed", "input_compressed", "both"]
INPUT_PRICE, OUTPUT_PRICE = 3.0, 15.0
MIN_REVIEWS, MAX_REVIEWS = 10, 20
MIN_AVG_LEN, MIN_TOTAL_LEN = 100, 2000


def subsample():
    DATA.mkdir(exist_ok=True)
    p = DATA / "reviews_150.json"
    if p.exists():
        return

    path = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023",
        "raw/review_categories/All_Beauty.jsonl", repo_type="dataset")
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    items = []
    for asin, sub in df.groupby("parent_asin"):
        n = len(sub)
        if n < MIN_REVIEWS or n > MAX_REVIEWS: continue
        avg_len = sub["text"].str.len().mean()
        total_len = sub["text"].str.len().sum()
        if avg_len < MIN_AVG_LEN or total_len < MIN_TOTAL_LEN: continue

        reviews = []
        for _, row in sub.iterrows():
            reviews.append({"rating": float(row["rating"]),
                            "title": row["title"], "text": row["text"]})
        items.append({"asin": asin, "n_reviews": int(n),
                      "total_chars": int(total_len), "reviews": reviews})

    items.sort(key=lambda x: x["total_chars"], reverse=True)
    items = items[:N]
    for i, item in enumerate(items):
        item["local_id"] = i + 1
    with open(p, "w") as f:
        json.dump(items, f, indent=2)


def format_reviews(item):
    parts = [f"[{r['rating']}/5] {r['title']}\n{r['text']}" for r in item["reviews"]]
    return "\n\n".join(parts)


def ask(client, text, system=None):
    msgs = [{"role": "user", "content": text}]
    kw = dict(model=MODEL, messages=msgs, temperature=0.0, max_tokens=2048)
    if system: kw["system"] = system

    for attempt in range(5):
        try:
            r = client.messages.create(**kw)
            return r.content[0].text, r.usage.input_tokens, r.usage.output_tokens
        except (anthropic.APITimeoutError, anthropic.APIConnectionError):
            if attempt == 4: raise
            time.sleep(2 ** attempt)


WORKERS = 10

def run_one(client, cond, item):
    raw = format_reviews(item)
    compressed = compress(raw) if cond in ("input_compressed", "both") else None
    text = compressed or raw
    prompt = f"Summarize these product reviews in plain text, no markdown:\n\n{text}"
    system = CAVEMAN if cond in ("output_compressed", "both") else None

    resp, inp, out = ask(client, prompt, system)
    return {"asin": item["asin"], "n_reviews": item["n_reviews"],
            "raw_input": raw, "compressed_input": compressed,
            "prompt": prompt, "response": resp,
            "input_tokens": inp, "output_tokens": out, "condition": cond}

def do_one(client, item, cond):
    d = OUT / f"product_{item['local_id']:03d}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{cond}.json"
    if p.exists(): return

    res = run_one(client, cond, item)
    tmp = tempfile.NamedTemporaryFile(mode="w", dir=d, suffix=".tmp", delete=False)
    json.dump(res, tmp, indent=2)
    tmp.close()
    os.rename(tmp.name, p)


def run_experiment():
    client = get_client()
    data = json.load(open(DATA / "reviews_150.json"))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [pool.submit(do_one, client, item, cond)
                for item in data for cond in CONDITIONS]
        for f in as_completed(jobs):
            f.result()


def bertscore_eval():
    from bert_score import score as bs_score
    dirs = sorted(OUT.glob("product_*"))

    for cond in CONDITIONS:
        refs, cands, paths = [], [], []
        for d in dirs:
            p = d / f"{cond}.json"
            if not p.exists(): continue
            r = json.load(open(p))
            if r.get("bertscore_f1") is not None: continue
            refs.append(r["raw_input"])
            cands.append(r["response"])
            paths.append(p)
        if not cands: continue

        _, _, f1 = bs_score(cands, refs, lang="en", verbose=True)
        for i, p in enumerate(paths):
            r = json.load(open(p))
            r["bertscore_f1"] = round(f1[i].item(), 4)
            tmp = tempfile.NamedTemporaryFile(mode="w", dir=p.parent, suffix=".tmp", delete=False)
            json.dump(r, tmp, indent=2)
            tmp.close()
            os.rename(tmp.name, p)


def analyze():
    dirs = sorted(OUT.glob("product_*"))
    rows = {}
    for cond in CONDITIONS:
        res = [json.load(open(d / f"{cond}.json"))
               for d in dirs if (d / f"{cond}.json").exists()]
        if not res: continue

        inp = [r["input_tokens"] for r in res]
        out = [r["output_tokens"] for r in res]
        bs = [r["bertscore_f1"] for r in res if r.get("bertscore_f1") is not None]
        total_in, total_out = sum(inp), sum(out)

        rows[cond] = {
            "mean_input": total_in / len(inp),
            "mean_output": total_out / len(out),
            "total_input": total_in, "total_output": total_out,
            "n": len(res),
            "mean_bertscore_f1": round(sum(bs) / len(bs), 4) if bs else None,
            "cost": (total_in * INPUT_PRICE + total_out * OUTPUT_PRICE) / 1e6,
        }

    with open(OUT / "analysis.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    subsample()
    run_experiment()
    bertscore_eval()
    analyze()
