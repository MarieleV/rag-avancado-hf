"""Funções puras do pipeline RAG (sem dependência de modelos).

Usadas tanto no notebook quanto no app do Hugging Face Spaces.
"""
import math
import re
import unicodedata

# ---------------------------------------------------------------- chunking
def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Divide o texto em pedaços de até `size` caracteres, respeitando fim de frase
    sempre que possível e repetindo ~`overlap` caracteres entre pedaços vizinhos."""
    text = " ".join(str(text).split())
    if len(text) <= size:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= size:
            cur = f"{cur} {s}".strip()
            continue
        if cur:
            chunks.append(cur)
            tail = cur[-overlap:]
            tail = tail.split(" ", 1)[-1] if " " in tail else tail  # começa em palavra inteira
            cur = f"{tail} {s}".strip()
        else:
            cur = s
        while len(cur) > size:  # frase gigante: corte duro com sobreposição
            chunks.append(cur[:size])
            cur = cur[size - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------ tokenização BM25
def _strip_accents(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))

_STOP_RAW = """a o e é de da do das dos em no na nos nas um uma uns umas para pra por pelo pela
pelos pelas com sem que se ao aos à às ou como mais mas foi foram ser era eram seu sua seus
suas ele ela eles elas isso isto esse essa este esta aquele aquela qual quais quem onde quando
também já não sim há tem têm entre sobre até após muito muitos muita muitas me te lhe nós vós"""
STOPWORDS = {_strip_accents(w) for w in _STOP_RAW.split()}

def tokenize(text: str) -> list[str]:
    """minúsculas + remoção de acentos + remoção de stopwords (pt-BR)."""
    t = _strip_accents(str(text).lower())
    return [w for w in re.findall(r"\w+", t) if len(w) > 1 and w not in STOPWORDS]


# ------------------------------------------------------------------- fusão
def rrf_fuse(rankings: list[list[int]], k: int = 60, top_n: int | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = soma 1/(k + posição). Recebe listas de ids
    (ordenadas da melhor para a pior) e devolve [(id, score)] ordenado."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_n] if top_n else fused


def to_doc_ranking(chunk_ids: list[int], chunk_doc_ids: list[str]) -> list[str]:
    """Converte ranking de chunks em ranking de documentos (mantém a 1ª aparição)."""
    seen, docs = set(), []
    for cid in chunk_ids:
        d = chunk_doc_ids[cid]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


# ----------------------------------------------------------------- métricas
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

def mrr_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0

def evaluate(runs: dict[str, list[str]], qrels: dict[str, set[str]], ks=(1, 3, 5, 10)) -> dict[str, float]:
    """runs: {query_id: [doc_ids ranqueados]} → médias das métricas."""
    out = {f"Recall@{k}": 0.0 for k in ks}
    out.update({"MRR@10": 0.0, "nDCG@10": 0.0})
    n = 0
    for qid, rel in qrels.items():
        ranked = runs.get(qid, [])
        for k in ks:
            out[f"Recall@{k}"] += recall_at_k(ranked, rel, k)
        out["MRR@10"] += mrr_at_k(ranked, rel)
        out["nDCG@10"] += ndcg_at_k(ranked, rel)
        n += 1
    return {m: round(v / max(n, 1), 4) for m, v in out.items()}


# ----------------------------------------------------------------- prompts
SYSTEM_PROMPT = (
    "Você é um assistente que responde perguntas em português do Brasil usando APENAS "
    "os trechos de contexto fornecidos. Regras:\n"
    "1. Cite as fontes entre colchetes, como [1] ou [2][3], logo após cada afirmação.\n"
    "2. Se o contexto não contiver a resposta, diga exatamente: "
    "\"Não encontrei essa informação nos documentos.\"\n"
    "3. Seja direto: no máximo 4 frases."
)

def build_messages(question: str, passages: list[str]) -> list[dict]:
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n{ctx}\n\nPergunta: {question}\n\nResposta:"},
    ]

HYDE_PROMPT = (
    "Escreva um parágrafo curto (3 a 4 frases), em estilo de enciclopédia e em português, "
    "que responda à pergunta abaixo. Se não souber, invente um texto plausível — ele será "
    "usado apenas para busca semântica.\n\nPergunta: {q}\n\nParágrafo:"
)
