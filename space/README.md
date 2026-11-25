---
title: RAG Avançado PT-BR
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Busca híbrida + reranking + HyDE sobre Wikipédia em português
---

# RAG Avançado — Busca Híbrida, Reranking e HyDE

Projeto do Módulo 2 (RAG Avançado) — Machine Learning / Deep Learning, Católica de Santa Catarina.

**Pipeline:** chunking → embeddings `BAAI/bge-m3` + FAISS → BM25 → fusão RRF → reranking com
`BAAI/bge-reranker-v2-m3` → (opcional) HyDE → LLM open-source via Inference Providers, com citações.

**Dados:** `hakari-bench/NanoBEIR-pt` (subconjunto NanoNQ).

## Configuração do Space
| Nome | Tipo | Valor |
|---|---|---|
| `HF_TOKEN` | Secret | token com permissão *Inference Providers* |
| `INDEX_REPO` | Variable | `mariele-vieira/rag-nanonq-pt-index` (gerado pelo notebook) |
| `LLM_MODELS` | Variable (opcional) | modelos separados por vírgula, tentados em ordem |
