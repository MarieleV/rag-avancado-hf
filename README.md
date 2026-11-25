# RAG Avançado com Hugging Face

**Busca híbrida, reranking e HyDE: medindo quanto cada técnica melhora a recuperação em português.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1LkVumJ6Iie85-qsgdWIetZetARjoUJKx/view?usp=sharing)
[![Demo no Colab](https://img.shields.io/badge/Demo-Gradio%20no%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/github/MarieleV/rag-avancado-hf/blob/main/notebooks/Demo_Gradio_Colab.ipynb)
[![Índice no Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-rag--nanonq--pt--index-blue)](https://huggingface.co/datasets/mariele-vieira/rag-nanonq-pt-index)

Projeto de avaliação final da disciplina **Machine Learning — Deep Learning**

(Prof. Claudinei Dias · Católica de Santa Catarina), **Módulo 2: Arquiteturas de Recuperação de Informação
(RAG Avançado e GraphRAG)**.

**Alunos:** Mariele Vieira, Maria A. Giuliari e Alexandre Sebastian 

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
> (o tempo total aparece na saída `%%time` da seção 9 do notebook). No experimento, gerar os 50 documentos hipotéticos levou 7 min 48 s (≈ 9,4 s por pergunta, contra ≈ 0,27 s de toda a recuperação).


![Resultados](results/resultados_rag.png)

## Estrutura

```
├── notebooks/RAG_Avancado_NanoNQ.ipynb   # experimento completo (Colab, GPU T4)
├── notebooks/Demo_Gradio_Colab.ipynb     # sobe a interface Gradio com link público
├── src/rag_core.py                       # chunking, BM25, RRF, métricas, prompts
├── space/                                # app Gradio (roda no Colab ou num Space)
├── tests/test_rag_core.py                # testes das funções puras
├── results/                              # gráficos e JSON gerados pelo notebook
└── docs/relatorio.md                      # relatório do projeto
```

## Como executar

**Notebook:** clique no badge "Open in Colab", escolha *Ambiente de execução → T4 GPU* e execute todas as células
(≈ 20–30 min). Ao final, ele salva `resultados_rag.png` e `resultados_rag.json` — copie-os para `results/`.

**Testes:**

```bash
pip install pytest
pytest tests/
```

## Demo interativa no Colab

A interface Gradio (pasta `space/`) permite fazer perguntas, alternar entre RAG naive, híbrido e híbrido + reranking,
ligar o HyDE e comparar lado a lado as fontes recuperadas.

**Por que no Colab e não num Space?** O Hugging Face passou a exigir o plano PRO para Spaces com SDK Gradio ou
Docker; na conta gratuita só há Spaces estáticos, que não rodam os modelos em PyTorch. Por isso a demo roda no
Colab, com GPU T4 gratuita e um link público temporário do Gradio.

1. Abra o badge **"Demo no Colab"** e escolha *Ambiente de execução → T4 GPU*.
2. *Executar tudo* → abra o link `https://….gradio.live` que aparece na última célula.

A demo não recalcula nada: ela baixa o índice já publicado no Hub
([`mariele-vieira/rag-nanonq-pt-index`](https://huggingface.co/datasets/mariele-vieira/rag-nanonq-pt-index),
seção 13 do notebook) e roda o LLM gerador (`Qwen3-4B-Instruct-2507`, 4 bits) na própria GPU do Colab, sem
precisar de token nem de créditos do *Inference Providers*. O link funciona enquanto a sessão do Colab estiver ativa.

> **Num Space (conta PRO):** crie um Space com SDK Gradio, envie os arquivos de `space/` e defina em *Settings* o
> secret `HF_TOKEN` e a variável `INDEX_REPO = mariele-vieira/rag-nanonq-pt-index`.
>
> `space/rag_core.py` é uma cópia de `src/rag_core.py` (o app precisa do arquivo na mesma pasta).
> Ao alterar um, copie para o outro.

## Referências

- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE).
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*.
- Chen et al. (2024). *BGE M3-Embedding*.
- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*.


Licença MIT.
