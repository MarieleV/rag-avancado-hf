# RAG Avançado com Hugging Face

**Busca híbrida, reranking e HyDE: medindo quanto cada técnica melhora a recuperação em português.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1LkVumJ6Iie85-qsgdWIetZetARjoUJKx/view?usp=sharing)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-demo-blue)](https://huggingface.co/spaces/SEU-USUARIO/rag-avancado-pt)

Projeto de avaliação final da disciplina **Machine Learning — Deep Learning**
(Prof. Claudinei Dias · Católica de Santa Catarina), **Módulo 2: Arquiteturas de Recuperação de Informação
(RAG Avançado e GraphRAG)**.

## O que o projeto faz

Constrói um pipeline de *Retrieval-Augmented Generation* só com modelos abertos do Hugging Face Hub e compara,
com gabarito, cinco configurações de recuperação:

| # | Configuração | Técnica |
|---|---|---|
| 1 | Denso (naive) | `BAAI/bge-m3` + FAISS |
| 2 | BM25 | busca léxica (`rank_bm25`) |
| 3 | Híbrido | BM25 + denso fundidos por Reciprocal Rank Fusion |
| 4 | Híbrido + Rerank | cross-encoder `BAAI/bge-reranker-v2-m3` nos 30 melhores |
| 5 | HyDE + Híbrido + Rerank | busca densa com resposta hipotética gerada pelo LLM |

A resposta final é gerada por `Qwen/Qwen3-4B-Instruct-2507` (4 bits), citando as fontes `[n]`.
O notebook inclui ainda uma demonstração em miniatura de **GraphRAG** (grafo de entidades + comunidades).

```
Pergunta ─► [HyDE] ─► busca densa (top-50) ─┐
        └──────────► BM25 (top-50) ─────────┴─► RRF (top-30) ─► reranker (top-5) ─► LLM ─► resposta [n]
```

## Dados

[`hakari-bench/NanoBEIR-pt`](https://huggingface.co/datasets/hakari-bench/NanoBEIR-pt), subconjunto **NanoNQ**:
5.035 passagens da Wikipédia em português, 50 perguntas e o gabarito das passagens relevantes.

## Resultados

| Configuração | Recall@5 | MRR@10 | nDCG@10 | ms/consulta |
|---|---|---|---|---|
| 1. Denso (naive) | 0.64 | 0.548 | 0.574 | 3.0 |
| 2. BM25 | 0.43 | 0.298 | 0.357 | 7.9 |
| 3. Híbrido (RRF) | 0.61 | 0.444 | 0.502 | 10.8 |
| 4. Híbrido + Rerank | 0.70 | 0.634 | 0.653 | 235.2 |
| 5. HyDE + Híbrido + Rerank | 0.72 | 0.625 | 0.649 | 260.4 |

> **Sobre a latência:** a coluna mede só a **recuperação** (busca densa, BM25, RRF e reranking).
> Na linha 5, o tempo de **geração do documento hipotético pelo LLM (HyDE) não está incluído** — ele é uma
> chamada extra ao LLM por pergunta e costuma custar mais do que todas as outras etapas juntas
> (o tempo total aparece na saída `%%time` da seção 9 do notebook).

![Resultados](results/resultados_rag.png)

## Estrutura

```
├── notebooks/RAG_Avancado_NanoNQ.ipynb   # experimento completo (Colab, GPU T4)
├── src/rag_core.py                       # chunking, BM25, RRF, métricas, prompts
├── space/                                # app Gradio para o Hugging Face Spaces
├── tests/test_rag_core.py                # testes das funções puras
├── results/                              # gráficos e JSON gerados pelo notebook
└── docs/                                 # relatório e slides (PDF)
```

## Como executar

**Notebook:** clique no badge "Open in Colab", escolha *Ambiente de execução → T4 GPU* e execute todas as células
(≈ 20–30 min). Ao final, ele salva `resultados_rag.png` e `resultados_rag.json` — copie-os para `results/`.

**Testes:**

```bash
pip install pytest
pytest tests/
```

**Space:** crie um Space (SDK Gradio), envie o conteúdo de `space/` e configure em *Settings*:

| Nome | Tipo | Valor |
|---|---|---|
| `HF_TOKEN` | Secret | token com permissão *Inference Providers* |
| `INDEX_REPO` | Variable | dataset publicado pelo notebook (seção 13) |

> `space/rag_core.py` é uma cópia de `src/rag_core.py` (o Space precisa do arquivo na mesma pasta).
> Ao alterar um, copie para o outro.

## Referências

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE).
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*.
- Chen et al. (2024). *BGE M3-Embedding*.
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.


Licença MIT.
