"""RAG Avançado (Busca Híbrida + Reranking + HyDE) — interface Gradio.

Executada pelo notebook notebooks/Demo_Gradio_Colab.ipynb (GPU T4 do Colab, link público).
Também funciona num Hugging Face Space com SDK Gradio (exige conta PRO).

Variáveis de ambiente:
  INDEX_REPO   dataset com chunks + embeddings publicado pelo notebook (seção 13),
               ex.: mariele-vieira/rag-nanonq-pt-index. Se ausente, o índice é construído na inicialização.
  LLM_LOCAL    "1" roda o LLM na própria GPU, em 4 bits (modelo: LOCAL_LLM_MODEL). Usado na demo do Colab.
  HF_TOKEN     token com permissão "Inference Providers"; só é usado quando o LLM não roda localmente.
  LLM_MODELS   modelos do Inference Providers, separados por vírgula, tentados em ordem.
  GRADIO_SHARE "1" cria um link público temporário (.gradio.live).
"""
import os
import sys
import time
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# As funções puras ficam em src/rag_core.py (mesmo código testado em tests/ e usado no notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_core import HYDE_PROMPT, build_messages, chunk_text, rrf_fuse, tokenize  # noqa: E402

DATASET_ID = os.getenv("DATASET_ID", "hakari-bench/NanoBEIR-pt")
SPLIT = os.getenv("SPLIT", "NanoNQ")
INDEX_REPO = os.getenv("INDEX_REPO", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODELS = [m.strip() for m in os.getenv(
    "LLM_MODELS", "Qwen/Qwen3-4B-Instruct-2507,openai/gpt-oss-20b,meta-llama/Llama-3.1-8B-Instruct").split(",") if m.strip()]
K_CANDIDATES, K_RERANK = 50, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODE_NAIVE = "RAG naive (só denso)"
MODE_HYBRID = "Híbrido (BM25 + denso, RRF)"
MODE_RERANK = "Híbrido + Reranking (cross-encoder)"

# ------------------------------------------------------------------ carga
print(f"Carregando modelos de embedding e reranking ({DEVICE})…")
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)

if INDEX_REPO:
    print(f"Baixando índice pré-calculado de {INDEX_REPO}…")
    idx_ds = load_dataset(INDEX_REPO, split="train")
    chunk_texts, chunk_doc_ids = list(idx_ds["text"]), [str(d) for d in idx_ds["doc_id"]]
    chunk_emb = np.asarray(idx_ds["embedding"], dtype="float32")
else:
    print("INDEX_REPO não definido: construindo índice a partir do corpus (pode levar alguns minutos)…")
    corpus = load_dataset(DATASET_ID, "corpus", split=SPLIT)
    chunk_texts, chunk_doc_ids = [], []
    for did, text in zip(corpus["_id"], corpus["text"]):
        for ch in chunk_text(text):
            chunk_texts.append(ch)
            chunk_doc_ids.append(str(did))
    chunk_emb = embedder.encode(chunk_texts, batch_size=32, normalize_embeddings=True,
                                show_progress_bar=True, convert_to_numpy=True).astype("float32")

faiss.normalize_L2(chunk_emb)
index = faiss.IndexFlatIP(chunk_emb.shape[1])
index.add(chunk_emb)
bm25 = BM25Okapi([tokenize(t) for t in chunk_texts])
print(f"Índice pronto: {len(chunk_texts):,} chunks")

try:  # perguntas do benchmark como exemplos
    EXAMPLES = [q for q in load_dataset(DATASET_ID, "queries", split=SPLIT)["text"]][:8]
except Exception:
    EXAMPLES = []

client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
local_tok = local_llm = None
if os.getenv("LLM_LOCAL", "").lower() in ("1", "true"):
    if DEVICE == "cuda":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f"Carregando LLM local {LOCAL_MODEL} (4 bits)…")
        local_tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        local_llm = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16))
    else:
        print("LLM_LOCAL ignorado: sem GPU. Usando Inference Providers.")


# ------------------------------------------------------------------ LLM
def local_chat(messages, max_tokens):
    prompt = local_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = local_tok(prompt, return_tensors="pt").to(local_llm.device)
    with torch.no_grad():
        out = local_llm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 pad_token_id=local_tok.eos_token_id)
    return local_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def llm_chat(messages, max_tokens=400):
    """LLM local (se carregado) ou modelos do Inference Providers em ordem; devolve (texto, modelo) ou (None, erro)."""
    if local_llm is not None:
        try:
            text = local_chat(messages, max_tokens)
            if text:
                return text, f"{LOCAL_MODEL} (local, 4 bits)"
            return None, f"{LOCAL_MODEL}: resposta vazia"
        except Exception as e:
            return None, f"{LOCAL_MODEL} (local): {e}"
    last_err = "HF_TOKEN não configurado" if not os.getenv("HF_TOKEN") else ""
    for model in LLM_MODELS:
        try:
            out = client.chat.completions.create(model=model, messages=messages,
                                                 max_tokens=max_tokens, temperature=0.2)
            text = (out.choices[0].message.content or "").strip()
            if text:
                return text, model
        except Exception as e:  # modelo indisponível, limite de uso etc.
            last_err = f"{model}: {e}"
    return None, last_err


# ------------------------------------------------------------------ recuperação
def dense_search(text, k=K_CANDIDATES):
    q = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q, k)
    return ids[0].tolist(), scores[0].tolist()


def bm25_search(text, k=K_CANDIDATES):
    s = bm25.get_scores(tokenize(text))
    top = np.argsort(-s)[:k]
    return top.tolist(), s[top].tolist()


def retrieve(question, mode, k, use_hyde):
    t = {}
    info = {}
    t0 = time.time()
    dense_query, hypo = question, ""
    if use_hyde and mode != MODE_NAIVE:
        hypo, _ = llm_chat([{"role": "user", "content": HYDE_PROMPT.format(q=question)}], 160)
        if hypo:
            dense_query = f"{question}\n{hypo}"
        t["HyDE (LLM)"] = time.time() - t0
    t1 = time.time()
    d_ids, d_scores = dense_search(dense_query)
    t["Busca densa"] = time.time() - t1
    info["dense_pos"] = {c: i + 1 for i, c in enumerate(d_ids)}

    if mode == MODE_NAIVE:
        ranked = d_ids
        scores = dict(zip(d_ids, d_scores))
    else:
        t1 = time.time()
        b_ids, _ = bm25_search(question)
        info["bm25_pos"] = {c: i + 1 for i, c in enumerate(b_ids)}
        fused = rrf_fuse([b_ids, d_ids], k=60, top_n=K_CANDIDATES)
        t["BM25 + RRF"] = time.time() - t1
        ranked = [c for c, _ in fused]
        scores = dict(fused)
        if mode == MODE_RERANK:
            t1 = time.time()
            cands = ranked[:K_RERANK]
            rs = reranker.predict([(question, chunk_texts[c]) for c in cands], batch_size=16)
            order = np.argsort(-rs)
            ranked = [cands[j] for j in order]
            scores = {cands[j]: float(rs[j]) for j in order}
            t["Reranking"] = time.time() - t1

    picked, seen = [], set()  # um chunk por documento
    for c in ranked:
        if chunk_doc_ids[c] not in seen:
            seen.add(chunk_doc_ids[c])
            picked.append(c)
        if len(picked) == k:
            break
    return picked, scores, info, t, hypo


def answer(question, mode, k, use_hyde):
    question = (question or "").strip()
    if not question:
        return "Digite uma pergunta.", pd.DataFrame(), ""
    picked, scores, info, t, hypo = retrieve(question, mode, int(k), use_hyde)

    t1 = time.time()
    text, model = llm_chat(build_messages(question, [chunk_texts[c] for c in picked]))
    t["Geração (LLM)"] = time.time() - t1
    if text is None:
        text = f"⚠️ Não foi possível chamar o LLM ({model}). Veja abaixo os trechos recuperados."
        model = "—"

    rows = []
    for i, c in enumerate(picked, 1):
        rows.append({
            "#": f"[{i}]",
            "doc_id": chunk_doc_ids[c],
            "pos. denso": info["dense_pos"].get(c, ">50"),
            "pos. BM25": info.get("bm25_pos", {}).get(c, "—" if mode == MODE_NAIVE else ">50"),
            "score final": round(scores.get(c, 0.0), 4),
            "trecho": chunk_texts[c],
        })
    timing = " · ".join(f"{k}: {v*1000:.0f} ms" for k, v in t.items())
    meta = f"**Modo:** {mode} · **LLM:** `{model}`  \n⏱️ {timing}"
    if use_hyde and mode == MODE_NAIVE:
        meta += "\n\nℹ️ HyDE não foi aplicado: no modo RAG naive a busca usa só a pergunta original."
    if hypo:
        meta += f"\n\n<details><summary>Documento hipotético (HyDE)</summary>\n\n{hypo}\n</details>"
    return text, pd.DataFrame(rows), meta


def compare(question, k):
    a1, df1, m1 = answer(question, MODE_NAIVE, k, False)
    a2, df2, m2 = answer(question, MODE_RERANK, k, False)
    return a1, df1[["#", "doc_id", "trecho"]] if len(df1) else df1, a2, df2[["#", "doc_id", "trecho"]] if len(df2) else df2


# ------------------------------------------------------------------ interface
with gr.Blocks(title="RAG Avançado — Hugging Face") as demo:
    gr.Markdown(
        "# 🔎 RAG Avançado: busca híbrida + reranking + HyDE\n"
        f"Base: **{DATASET_ID} / {SPLIT}** ({len(chunk_texts):,} trechos da Wikipédia em português) · "
        f"Embeddings `{EMBED_MODEL}` · Reranker `{RERANK_MODEL}`"
    )
    with gr.Tab("Perguntar"):
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="Pergunta", placeholder="Ex.: Quem escreveu Dom Casmurro?", lines=2)
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=q, label="Perguntas do benchmark")
            with gr.Column(scale=2):
                mode = gr.Radio([MODE_NAIVE, MODE_HYBRID, MODE_RERANK], value=MODE_RERANK, label="Pipeline")
                hyde = gr.Checkbox(label="Query transformation (HyDE)", value=False,
                                   info="Disponível apenas nos modos híbridos (não se aplica ao RAG naive).")
                k = gr.Slider(1, 10, value=5, step=1, label="Trechos no contexto (k)")
                btn = gr.Button("Responder", variant="primary")
        out = gr.Markdown(label="Resposta")
        meta = gr.Markdown()
        table = gr.Dataframe(label="Fontes recuperadas", wrap=True)
        mode.change(lambda m: gr.update(interactive=(m != MODE_NAIVE)), mode, hyde)
        btn.click(answer, [q, mode, k, hyde], [out, table, meta])
        q.submit(answer, [q, mode, k, hyde], [out, table, meta])

    with gr.Tab("Comparar naive × avançado"):
        q2 = gr.Textbox(label="Pergunta", lines=2)
        k2 = gr.Slider(1, 10, value=3, step=1, label="k")
        btn2 = gr.Button("Comparar", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### RAG naive (denso)")
                a_n = gr.Markdown()
                t_n = gr.Dataframe(wrap=True)
            with gr.Column():
                gr.Markdown("### Híbrido + Reranking")
                a_a = gr.Markdown()
                t_a = gr.Dataframe(wrap=True)
        btn2.click(compare, [q2, k2], [a_n, t_n, a_a, t_a])

    with gr.Tab("Como funciona"):
        gr.Markdown(
            "1. **Chunking** dos documentos (≤800 caracteres, 120 de sobreposição).\n"
            "2. **Busca densa**: embeddings `bge-m3` + índice FAISS (similaridade de cosseno).\n"
            "3. **BM25**: busca léxica por termos exatos.\n"
            "4. **RRF**: funde as duas listas pela posição — `Σ 1/(60 + posição)`.\n"
            "5. **Reranking**: cross-encoder `bge-reranker-v2-m3` reordena os 30 melhores.\n"
            "6. **HyDE** (opcional): o LLM escreve uma resposta hipotética, usada só para a busca densa.\n"
            "7. **Geração**: LLM open-source (local em 4 bits ou via Inference Providers) responde citando `[n]`."
        )

if __name__ == "__main__":
    demo.queue().launch(share=os.getenv("GRADIO_SHARE", "").lower() in ("1", "true"))
