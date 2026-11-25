# Guia de estudo: RAG Avançado com Hugging Face

Material de preparação para a apresentação. Cada seção explica **o que** o projeto faz, **como** o código faz e
**por que** foi feito assim, terminando com o que costuma ser perguntado. No fim há um roteiro de apresentação, uma
lista de perguntas prováveis com respostas e uma "cola" com os números principais.

**Arquivos de referência:**
[`notebooks/RAG_Avancado_NanoNQ.ipynb`](../notebooks/RAG_Avancado_NanoNQ.ipynb) (experimento) ·
[`src/rag_core.py`](../src/rag_core.py) (funções puras) ·
[`space/app.py`](../space/app.py) (interface) ·
[`relatorio.md`](relatorio.md) (texto completo)

---

## Sumário

1. [O problema: por que RAG existe](#1-o-problema-por-que-rag-existe)
2. [Visão geral do pipeline](#2-visão-geral-do-pipeline)
3. [Dados: NanoBEIR-pt / NanoNQ](#3-dados-nanobeir-pt--nanonq)
4. [Chunking](#4-chunking)
5. [Busca densa: embeddings + FAISS](#5-busca-densa-embeddings--faiss)
6. [Busca léxica: BM25](#6-busca-léxica-bm25)
7. [Fusão: Reciprocal Rank Fusion (RRF)](#7-fusão-reciprocal-rank-fusion-rrf)
8. [Reranking com cross-encoder](#8-reranking-com-cross-encoder)
9. [Query transformation: HyDE](#9-query-transformation-hyde)
10. [O LLM gerador e o prompt](#10-o-llm-gerador-e-o-prompt)
11. [Métricas de avaliação](#11-métricas-de-avaliação)
12. [Resultados: como ler e explicar](#12-resultados-como-ler-e-explicar)
13. [GraphRAG](#13-graphrag)
14. [A demo Gradio](#14-a-demo-gradio)
15. [Engenharia: organização, testes e reprodutibilidade](#15-engenharia-organização-testes-e-reprodutibilidade)
16. [Contexto histórico: GPTs de 2022 a 2026](#16-contexto-histórico-gpts-de-2022-a-2026)
17. [Limitações e trabalhos futuros](#17-limitações-e-trabalhos-futuros)
18. [Roteiro sugerido para a apresentação](#18-roteiro-sugerido-para-a-apresentação)
19. [Perguntas prováveis do professor (com respostas)](#19-perguntas-prováveis-do-professor-com-respostas)
20. [Glossário](#20-glossário)
21. [Cola: números para saber de cor](#21-cola-números-para-saber-de-cor)

---

## 1. O problema: por que RAG existe

Um LLM sozinho só sabe o que memorizou no treino. Isso causa três problemas:

| Problema | Exemplo |
|---|---|
| **Conhecimento estático** | O modelo não sabe de nada que aconteceu depois da data de corte do treino. |
| **Alucinação** | Sem evidência, ele inventa uma resposta plausível com a mesma confiança de uma resposta certa. |
| **Sem rastreabilidade** | Não dá para saber de onde veio cada afirmação. |

**RAG (*Retrieval-Augmented Generation*, Lewis et al., 2020)** resolve isso juntando duas memórias:

- **Memória paramétrica:** o que está nos pesos do modelo (linguagem, raciocínio, conhecimento geral).
- **Memória não paramétrica:** um índice de documentos consultado **na hora da pergunta**.

Fluxo: a pergunta chega → o sistema **busca** os trechos mais relevantes → coloca esses trechos no prompt → o LLM
**responde apoiado neles e cita as fontes**.

> **Frase para a apresentação:** "RAG transforma o LLM de alguém que responde de memória em alguém que consulta
> documentos antes de responder, e mostra de onde tirou a resposta."

**RAG × fine-tuning** (pergunta clássica):

| | RAG | Fine-tuning |
|---|---|---|
| Atualizar conhecimento | Basta reindexar documentos | Precisa treinar de novo |
| Citar fontes | Sim | Não |
| Custo | Baixo (só inferência) | Alto (GPU para treino) |
| Melhor para | Fatos, documentos que mudam | Estilo, formato, tarefa específica |

Eles não são excludentes: dá para fazer fine-tuning de estilo e usar RAG para os fatos.

**Naive RAG × Advanced RAG:** o RAG "naive" faz só *chunking → embedding → top-k → LLM*. Ele falha quando a busca
traz trechos errados (o LLM responde com base em lixo) ou quando traz trechos demais (o LLM se perde). O *Advanced
RAG* adiciona técnicas **antes** da busca (transformar a pergunta), **na** busca (híbrida) e **depois** da busca
(reranking). Este projeto mede cada uma delas.

---

## 2. Visão geral do pipeline

O sistema tem duas fases.

**Indexação (feita uma vez):**

```text
Corpus (5.035 documentos)
   └─► chunking (6.643 trechos)
         ├─► bge-m3 ──► vetores de 1024 dimensões ──► índice FAISS     (busca densa)
         └─► tokenização ─────────────────────────► índice BM25        (busca léxica)
```

**Consulta (a cada pergunta):**

```text
Pergunta ─► [HyDE opcional] ─► busca densa (top-50) ─┐
        └─────────────────────► BM25 (top-50) ───────┴─► RRF ─► top-30 ─► cross-encoder ─► top-5 ─► LLM ─► resposta [n]
```

**As 5 configurações comparadas** (cada uma adiciona uma peça):

| # | Configuração | O que testa |
|---|---|---|
| 1 | Denso (naive) | O ponto de partida: só embeddings |
| 2 | BM25 | Só busca por palavras exatas |
| 3 | Híbrido (RRF) | Juntar as duas buscas ajuda? |
| 4 | Híbrido + Rerank | O cross-encoder reordenando ajuda? |
| 5 | HyDE + Híbrido + Rerank | Transformar a pergunta ajuda? |

**Constantes do experimento** (célula de configuração do notebook):

| Constante | Valor | Significado |
|---|---|---|
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 800, 120 | Tamanho máximo do trecho e sobreposição, em caracteres |
| `K_CANDIDATES` | 50 | Quantos candidatos cada buscador devolve |
| `K_RERANK` | 30 | Quantos candidatos vão para o cross-encoder |
| `K_CONTEXT` | 5 | Quantos trechos entram no prompt do LLM |
| `SEED` | 42 | Semente para reprodutibilidade |

---

## 3. Dados: NanoBEIR-pt / NanoNQ

- **Dataset:** [`hakari-bench/NanoBEIR-pt`](https://huggingface.co/datasets/hakari-bench/NanoBEIR-pt), subconjunto
  **NanoNQ**.
- **BEIR** é o benchmark padrão de recuperação de informação (Thakur et al., 2021). **NanoBEIR** são versões
  reduzidas dele, feitas para avaliar rápido. **NQ** é o *Natural Questions* (Google): perguntas reais de busca com
  respostas na Wikipédia. A versão **-pt** foi traduzida automaticamente para português.
- **Três partes:**

| Parte | Conteúdo | No código |
|---|---|---|
| `corpus` | 5.035 passagens da Wikipédia | `doc_ids`, `doc_texts` |
| `queries` | 50 perguntas | `queries = {id: texto}` |
| `qrels` | Gabarito: quais passagens respondem cada pergunta | `qrels = {query_id: {doc_ids relevantes}}` |

**Por que este dataset?** Porque tem **gabarito** (*qrels*). Sem gabarito, só daria para mostrar exemplos que
"parecem bons". Com gabarito, dá para **medir** se cada técnica melhora ou piora, com números.

**Cuidados para mencionar:** são só 50 perguntas (amostra pequena) e o texto é uma **tradução automática**, o que
pode prejudicar a busca por palavras exatas (pergunta e passagem foram traduzidas separadamente e podem usar palavras
diferentes para a mesma coisa).

---

## 4. Chunking

**O que é:** dividir cada documento em pedaços menores (*chunks*) antes de indexar.

**Por que precisa:**
1. Modelos de embedding têm limite de tokens, e um vetor que representa um texto longo "dilui" os detalhes.
2. O LLM recebe só alguns trechos: pedaços menores e focados desperdiçam menos contexto.

**Como o código faz** (`chunk_text` em `src/rag_core.py`):

1. Normaliza os espaços.
2. Se o texto tem até 800 caracteres, vira um chunk só.
3. Senão, quebra em **frases** (regex que corta depois de `.`, `!` ou `?`) e vai juntando frases enquanto couber em
   800 caracteres.
4. Ao fechar um chunk, o próximo começa com os **últimos ~120 caracteres** do anterior (a **sobreposição**), cortados
   no início de uma palavra inteira.
5. Se uma única frase passa de 800 caracteres, faz um corte "duro" com sobreposição.

**Por que sobreposição?** Se uma informação cai na fronteira entre dois chunks (metade em cada), a sobreposição
garante que pelo menos um deles tenha a informação inteira.

**Por que respeitar frases?** Cortar no meio da frase gera trechos sem sentido, que dão embeddings piores.

**Cada chunk guarda o `doc_id` de origem.** Isso serve para (a) citar a fonte e (b) avaliar no nível de documento.

**Resultado:** 5.035 documentos → **6.643 chunks** (≈ 1,3 chunk por documento; a maioria das passagens já é curta).

> **Se perguntarem "por que 800?":** é um valor intermediário comum: grande o bastante para conter um fato com
> contexto, pequeno o bastante para 5 trechos caberem no prompt. Não fizemos busca de hiperparâmetro; isso seria
> um trabalho futuro.

---

## 5. Busca densa: embeddings + FAISS

### Embedding

Um **embedding** transforma um texto num vetor de números em que **textos com significado parecido ficam perto**.
"Quem escreveu Dom Casmurro?" fica perto de "Machado de Assis é o autor de Dom Casmurro", mesmo sem muitas palavras
em comum.

**Modelo: `BAAI/bge-m3`**
- Multilíngue (100+ idiomas, bom em português).
- Vetores de **1024 dimensões**; aceita até **8.192 tokens**.
- Treinado com pares pergunta–passagem (aprendizado contrastivo), então é feito para busca.
- Usado via `sentence-transformers`, em `float16` na GPU.

### Bi-encoder

O bge-m3 é um **bi-encoder**: codifica a pergunta e o documento **separadamente**. A grande vantagem é que os
vetores dos documentos podem ser **calculados uma vez** e guardados. Na hora da pergunta, só a pergunta é codificada.

### Similaridade de cosseno e normalização

A proximidade entre vetores é medida pelo **cosseno do ângulo** entre eles. Os vetores são **normalizados**
(comprimento 1, `normalize_embeddings=True`). Com vetores de comprimento 1, o **produto interno é igual ao
cosseno**, e por isso o índice usado é o `IndexFlatIP` (IP = *inner product*).

### FAISS

**FAISS** (Facebook AI Similarity Search) é uma biblioteca de busca vetorial.
- `IndexFlatIP` é busca **exata** (força bruta): compara a pergunta com todos os 6.643 vetores. Com esse tamanho,
  leva poucos milissegundos.
- Para milhões de vetores, usaríamos um índice **aproximado** (IVF, HNSW), trocando um pouco de precisão por
  velocidade.
- O índice é **salvo em disco** (`faiss.write_index`) e recarregado para provar que funciona, como pede o enunciado
  ("armazenar os vetores localmente").

**Por que FAISS e não Chroma?** O enunciado aceita os dois. FAISS é mais leve, roda dentro do notebook sem servidor
e, para um índice deste tamanho, não precisamos dos recursos de banco de dados (metadados, filtros) do Chroma.

### Onde o denso erra

1. **Termos exatos:** nomes próprios, números e siglas raros podem ser "diluídos" no vetor.
2. **Ruído:** trechos sobre o mesmo assunto, mas sem a resposta, ficam perto da pergunta e ocupam o topo.

A seção 5 do notebook mostra exemplos: em **15 das 50 perguntas**, a passagem correta não apareceu no top-5 denso.

---

## 6. Busca léxica: BM25

**O que é:** um algoritmo clássico (Robertson e Zaragoza, 2009) que pontua documentos pelas **palavras exatas** em
comum com a pergunta. É o que buscadores tradicionais usam.

**Intuição da fórmula** (não precisa decorar a fórmula, mas sim estas três ideias):

1. **Frequência do termo (TF) com saturação:** a palavra aparecer mais vezes no documento aumenta o score, mas com
   retorno decrescente (aparecer 10 vezes não vale 10× mais que 1 vez). Parâmetro `k1` (padrão 1,5).
2. **Raridade do termo (IDF):** palavras raras no corpus valem mais. "Voltaire" vale muito mais que "história".
3. **Normalização pelo tamanho:** documentos longos têm mais palavras por acaso, então são penalizados.
   Parâmetro `b` (padrão 0,75).

**Tokenização** (`tokenize` em `src/rag_core.py`):
- minúsculas;
- remoção de acentos (`São` → `sao`), para "população" e "populacao" baterem;
- remoção de *stopwords* em português ("a", "de", "que", "qual"...), palavras que não ajudam a diferenciar documentos;
- descarte de tokens de uma letra.

Exemplo do teste: `tokenize("Qual é a população de São Paulo?")` → `["populacao", "sao", "paulo"]`.

**O que não fizemos:** *stemming* (reduzir "ferroviário" e "ferrovia" a um mesmo radical). Em português, que tem
muita flexão de palavras, isso pode ter enfraquecido o BM25.

**Por que o BM25 foi fraco aqui (Recall@5 = 0,43):**
- As perguntas do NQ são **linguagem natural**, muitas vezes parafraseadas em relação à passagem.
- A **tradução automática** pode usar palavras diferentes na pergunta e na passagem.
- Sem stemming, variações da mesma palavra não batem.

---

## 7. Fusão: Reciprocal Rank Fusion (RRF)

**Problema:** o BM25 dá scores como 12,7 e o denso dá cossenos como 0,63. As escalas são incompatíveis; somar ou
fazer média não faz sentido sem calibração.

**Solução do RRF (Cormack et al., 2009):** ignorar os scores e usar **só a posição** de cada documento em cada
lista:

$$\text{RRF}(d) = \sum_{\text{listas}} \frac{1}{k + \text{posição}(d)} \qquad k = 60$$

**Exemplo numérico:**

| Documento | Posição no denso | Posição no BM25 | Score RRF |
|---|---|---|---|
| A | 1 | não aparece | 1/61 = **0,0164** |
| B | 3 | 3 | 1/63 + 1/63 = **0,0317** |
| C | não aparece | 1 | 1/61 = **0,0164** |

O documento B, que aparece bem nas **duas** listas, ganha do A, que é o primeiro de uma lista só. O RRF premia a
**concordância** entre buscadores.

**Para que serve o k = 60?** Ele suaviza a diferença entre posições. Sem ele (k = 0), o 1º lugar valeria 1 e o
2º valeria 0,5: o topo de cada lista dominaria. Com k = 60, 1/61 e 1/62 são quase iguais. O valor 60 vem do artigo
original e é o padrão da área.

**Por que o híbrido piorou neste projeto:**
O RRF dá **peso igual** às duas listas. Como o BM25 é muito mais fraco aqui, o 1º lugar do BM25 (muitas vezes
errado) recebe o mesmo score que o 1º lugar do denso (muitas vezes certo): veja os documentos A e C na tabela.
Resultado: lixo do BM25 se intercala com os acertos do denso e empurra a passagem certa para baixo.

> **Lição:** busca híbrida não é ganho garantido. Ela funciona quando os dois buscadores têm qualidade parecida e
> erram em perguntas **diferentes**. Correções possíveis: RRF **ponderado** (peso maior para o denso) ou
> comparar com "denso + rerank" sem BM25.

---

## 8. Reranking com cross-encoder

### Bi-encoder × cross-encoder

| | Bi-encoder (bge-m3) | Cross-encoder (bge-reranker-v2-m3) |
|---|---|---|
| Entrada | Pergunta e trecho **separados** | Pergunta e trecho **juntos**: `[pergunta; trecho]` |
| Saída | Um vetor para cada texto | Um **score de relevância** para o par |
| Interação entre os textos | Nenhuma (só compara os vetores no fim) | **Atenção cruzada** entre todos os tokens da pergunta e do trecho |
| Pré-cálculo | Sim, os vetores ficam no índice | Não, precisa rodar para cada par |
| Precisão | Boa | **Muito melhor** |
| Custo | Barato | Caro |

**Por que não usar o cross-encoder no corpus inteiro?** Seriam 6.643 passagens pelo modelo **a cada pergunta**.
Com 30 candidatos, são só 30.

**Padrão *retrieve-then-rerank* (Nogueira e Cho, 2019):** um buscador rápido traz dezenas de candidatos com boa
**cobertura** (recall); o cross-encoder, lento e preciso, ajusta a **ordem** (precisão no topo).

**No código** (função `rerank` no notebook): monta pares `(pergunta, trecho)` para os 30 primeiros do RRF, chama
`reranker.predict`, ordena pelos scores.

**Por que ele foi o maior ganho:** a passagem certa geralmente **já estava** entre os 30 candidatos do RRF, só que
mal posicionada. O cross-encoder "lê" pergunta e trecho juntos e a traz para o topo. Ele também desfaz o estrago do
RRF: MRR@10 de 0,444 (híbrido) para **0,634**, e Recall@1 de 0,30 para **0,49**.

**Custo:** a latência de recuperação vai de ~11 ms para ~235 ms por pergunta (≈ 22×). Ainda é pouco perto da geração
do LLM (segundos), então compensa.

**Depois do rerank, "um trecho por documento":** ao escolher os 5 trechos do contexto, o código pega no máximo um
trecho de cada documento. Isso **diversifica** o contexto e evita gastar o prompt com pedaços repetidos do mesmo
texto.

---

## 9. Query transformation: HyDE

**HyDE = *Hypothetical Document Embeddings* (Gao et al., 2022).**

**O problema que ele ataca:** perguntas são **curtas e interrogativas** ("Quem faz o papel de Heather em A Bela e a
Fera?"); documentos são **longos e afirmativos**. Eles podem ficar em regiões diferentes do espaço vetorial.

**A ideia:**
1. Pedir ao LLM um parágrafo que **responderia** a pergunta, em estilo de enciclopédia (`HYDE_PROMPT`).
2. Buscar com o embedding de `pergunta + parágrafo hipotético`. Esse texto "parece um documento" e cai mais perto
   dos documentos reais.

**O parágrafo pode estar errado, e tudo bem**: ele serve só para a busca. Por isso:
- o **BM25 continua usando a pergunta original** (palavras inventadas atrapalhariam a busca exata);
- o **cross-encoder reordena com a pergunta original**;
- o texto hipotético **nunca vai para o prompt final** (senão o LLM poderia repetir o fato inventado).

**Resultado:** praticamente neutro. Recall@5 de 0,70 para 0,72 (uma pergunta a mais), MRR@10 de 0,634 para 0,625 e
nDCG@10 de 0,653 para 0,649. Tudo dentro do ruído.

**Por que não ajudou (bons argumentos para a banca):**
1. **O HyDE foi proposto para buscadores sem supervisão** (o artigo usa o Contriever, treinado sem pares
   pergunta–resposta). O **bge-m3 já foi treinado com pares pergunta–passagem**, então ele já sabe aproximar
   perguntas de respostas. Sobra pouco para o HyDE melhorar.
2. **LLM pequeno (4B):** o parágrafo hipotético às vezes traz fatos errados, que puxam a busca para o lugar errado.
   No notebook, o primeiro exemplo de HyDE tem um fato errado.
3. **O reranker neutraliza o efeito:** ele reordena com a pergunta original, então pequenas mudanças nos candidatos
   pesam pouco no resultado final.
4. **HyDE só mexe numa das duas listas** do RRF (a densa).

**Custo:** uma chamada extra ao LLM por pergunta: **7 min 48 s para 50 perguntas (~9,4 s cada)**, mais de 30×
o tempo de toda a recuperação. Esse tempo **não está** na coluna de latência da tabela (ela mede só a recuperação),
o que é dito explicitamente no README e no relatório.

> **Conclusão para falar:** "Nesta configuração, o HyDE não se paga: custa ~9 s por pergunta e não melhora a
> recuperação de forma mensurável."

---

## 10. O LLM gerador e o prompt

**Modelo: `Qwen/Qwen3-4B-Instruct-2507`**
- 4 bilhões de parâmetros, licença Apache 2.0 (aberto), bom em português.
- Versão *Instruct* de julho/2025 ("2507"), sem modo de "pensamento", com contexto longo (256K tokens).
- Usado em três funções: gerar a **resposta final**, escrever o **documento hipotético do HyDE** e extrair
  **triplas do GraphRAG**.

**Quantização em 4 bits (NF4, `bitsandbytes`):**
- Em vez de guardar cada peso com 16 bits, guarda com **4 bits**. O modelo cai de ~8 GB para **~3 GB de VRAM** e
  cabe na T4 gratuita do Colab (16 GB) junto com o embedder e o reranker.
- **NF4** (*NormalFloat 4*, do artigo QLoRA, Dettmers et al., 2023) distribui os 16 níveis possíveis de acordo com
  uma distribuição normal, que é como os pesos de redes neurais costumam se distribuir. Assim perde menos
  informação do que dividir em níveis iguais.
- Os cálculos são feitos em `float16` (`bnb_4bit_compute_dtype`).
- **Perde qualidade?** Um pouco, mas pequena para modelos deste tamanho; a troca vale para rodar de graça.

**Decodificação gulosa** (`do_sample=False`): o modelo sempre escolhe o token mais provável. Torna as respostas
**reprodutíveis** (mesma entrada, mesma saída), importante num experimento. `repetition_penalty=1.05` evita repetição.

**O prompt** (`SYSTEM_PROMPT` e `build_messages` em `src/rag_core.py`):
- "Responda **APENAS** com os trechos fornecidos." → reduz alucinação.
- "Cite as fontes entre colchetes, como [1]." → torna a resposta **verificável**.
- "Se o contexto não tiver a resposta, diga exatamente: *Não encontrei essa informação nos documentos.*" → recusa
  explícita em vez de inventar.
- "No máximo 4 frases." → respostas diretas.
- Os trechos entram numerados: `[1] ...`, `[2] ...`.

**Funciona sempre? Não** (bom ser honesto):
- No **teste de robustez** ("placar da final da Copa de 2026", fora da base), o modelo **recusou corretamente**.
- Na pergunta do **Anjo do Norte**, com só 3 trechos no contexto, a passagem certa ficou de fora e o pipeline
  avançado **respondeu com base em trechos de outro assunto**. A recusa não é garantida: o prompt reduz a
  alucinação, mas não a elimina.

---

## 11. Métricas de avaliação

As métricas são calculadas **no nível de documento**: o ranking de chunks é convertido em ranking de documentos
mantendo a primeira aparição de cada documento (`to_doc_ranking`). Motivo: o gabarito diz quais **documentos** são
relevantes, não quais chunks.

### Recall@k
**"Das passagens corretas, quantas apareceram entre as k primeiras?"**
- Exemplo: a pergunta tem 1 passagem correta e ela está na posição 4. Recall@5 = 1; Recall@3 = 0.
- Mede **cobertura**: se a resposta chega ao LLM. O Recall@5 importa muito porque 5 é o tamanho do contexto.

### MRR@10 (*Mean Reciprocal Rank*)
**"Quão perto do topo está a primeira passagem correta?"** Vale 1/posição.
- Posição 1 → 1; posição 2 → 0,5; posição 4 → 0,25; fora do top-10 → 0.
- A média nas 50 perguntas dá o MRR. Mede **ordem**: se a resposta está no topo.

### nDCG@10 (*normalized Discounted Cumulative Gain*)
**"Quantas corretas estão no topo, dando mais peso às primeiras posições?"**
- Cada acerto vale `1 / log2(posição + 1)`: posição 1 vale 1; posição 3 vale 0,5; posição 7 vale 0,33.
- Divide-se pelo valor ideal (todas as corretas no topo), então fica entre 0 e 1.
- Exemplo: 1 passagem correta, encontrada na posição 3 → nDCG = 0,5 / 1 = **0,5**.
- É a métrica padrão do BEIR e do MTEB.

**Diferença em uma frase:** Recall diz **se** achou; MRR diz **onde** achou a primeira; nDCG diz **onde** achou
todas, com desconto por posição.

---

## 12. Resultados: como ler e explicar

| Configuração | Recall@5 | MRR@10 | nDCG@10 | ms/consulta |
|---|:---:|:---:|:---:|:---:|
| 1. Denso (naive) | 0.64 | 0.548 | 0.574 | 3.0 |
| 2. BM25 | 0.43 | 0.298 | 0.357 | 7.9 |
| 3. Híbrido (RRF) | 0.61 | 0.444 | 0.502 | 10.8 |
| 4. Híbrido + Rerank | 0.70 | **0.634** | **0.653** | 235.2 |
| 5. HyDE + Híbrido + Rerank | **0.72** | 0.625 | 0.649 | 260.4 |

**Leitura linha a linha:**

1. **Denso:** bom ponto de partida (o bge-m3 é forte), mas em 15 das 50 perguntas a passagem certa ficou fora do
   top-5.
2. **BM25:** o mais fraco, pelos motivos da seção 6.
3. **Híbrido:** **piorou** em relação ao denso em todas as métricas (seção 7).
4. **Híbrido + Rerank:** **o maior ganho** do projeto. Melhor MRR e nDCG; Recall@1 de 0,49 (contra 0,39 do
   denso). Das 15 perguntas que o naive errou no top-5, **resgatou 5**.
5. **HyDE:** neutro (seção 9) e caro.

**Exemplos de perguntas "resgatadas" pelo rerank** (posição da passagem correta):

| Pergunta | Naive | Com rerank |
|---|:---:|:---:|
| Por que existe o Anjo do Norte? | 9 | 4 |
| Quais foram os resultados do desenvolvimento ferroviário na Flórida? | 15 | 3 |
| "Estou disposto a dar a vida para defender o seu direito de dizê-lo." | 8 | 1 |

Na comparação lado a lado com 3 trechos, na frase atribuída a Voltaire o naive respondeu "Não encontrei essa
informação", e o avançado explicou, citando `[1]`, que a frase é de Evelyn Beatrice Hall, frequentemente atribuída a
Voltaire.

**Sobre a latência:**
- Mede só a recuperação; o HyDE e a geração do LLM não entram.
- A busca densa roda **em lote** para as 50 perguntas, então os 3 ms por pergunta estão um pouco **subestimados** em
  relação a um uso real, pergunta a pergunta.
- Na demo, os tempos aparecem por etapa para cada pergunta.

> **As três mensagens principais:** (1) o reranker é o que mais vale a pena; (2) híbrido não é ganho automático;
> (3) o HyDE não se pagou com um LLM pequeno e um embedder já treinado para busca.

---

## 13. GraphRAG

**O problema:** o RAG por chunks responde bem a perguntas **locais** ("quem escreveu X?"), em que a resposta está
num trecho. Mas falha em perguntas **globais** ("quais são os temas principais destes documentos?"), porque a
resposta não está em nenhum trecho isolado.

**A proposta do GraphRAG (Edge et al., 2024, Microsoft):**
1. O LLM extrai **entidades e relações** de cada trecho → um **grafo de conhecimento**.
2. Um algoritmo de **detecção de comunidades** (Leiden ou Louvain) agrupa entidades muito conectadas entre si.
3. O LLM **resume cada comunidade**.
4. Perguntas globais são respondidas combinando os resumos (*map-reduce*).

**A versão em miniatura do notebook (seção 12):**
- Recupera 12 trechos para "história e geografia do Brasil".
- Pede ao LLM até 6 **triplas** por trecho, em JSON: `{"s": sujeito, "r": relação, "o": objeto}`.
- Monta o grafo com `networkx` e agrupa com **Louvain** (que maximiza a **modularidade**: muitas ligações dentro de
  cada grupo, poucas entre grupos).
- Resume as 4 maiores comunidades e responde a pergunta global a partir dos resumos.

**O que o grafo mostrou** (`results/graphrag_demo.png`):
- Comunidades com sentido: Copa de 1994 (Romário, Bebeto, Dunga), América Central (Nicarágua, Honduras,
  Guatemala...), história da geologia (Buffon, Werner), comércio de peles (Rio Columbia, Lewis e Clark).
- **Limitações visíveis:** os trechos recuperados são heterogêneos; a mesma entidade aparece duplicada ("Silva" e
  "silva"), porque não há **resolução de entidades**; o modelo de 4B cria entidades vagas ("ano desconhecido").

**Custo:** o GraphRAG chama o LLM para cada trecho na indexação, o que fica caro num corpus grande. Por isso surgiram
variantes leves como o LightRAG. Aqui foi uma **demonstração didática**, não uma avaliação.

---

## 14. A demo Gradio

**O que a interface faz** (`space/app.py`):
- Aba **Perguntar:** escolhe o pipeline (naive, híbrido, híbrido + rerank), liga ou desliga o HyDE, define k. Mostra
  a resposta, os tempos por etapa e uma tabela com cada trecho, sua posição no denso, no BM25 e o score final.
- Aba **Comparar naive × avançado:** a mesma pergunta nos dois pipelines, lado a lado.
- Aba **Como funciona:** resumo das etapas.

**Como ela roda** (`notebooks/Demo_Gradio_Colab.ipynb`):
1. Instala as dependências e clona o repositório do GitHub.
2. Baixa o **índice já publicado no Hub** (`mariele-vieira/rag-nanonq-pt-index`): trechos + embeddings em float16,
   publicados pela seção 13 do notebook. Assim não recalcula 6,6 mil embeddings.
3. Carrega o bge-m3, o reranker e o **Qwen3-4B em 4 bits na GPU do Colab** (`LLM_LOCAL=1`).
4. Sobe o Gradio com `share=True`, que gera um link público `https://….gradio.live`.

**Por que Colab e não Hugging Face Spaces?** Durante o projeto, o Hugging Face passou a exigir o plano PRO para
Spaces com Gradio, e os créditos gratuitos do *Inference Providers* (API para chamar LLMs) acabaram (erro 402). A
solução foi rodar tudo no Colab, com o LLM local. O app continua compatível com um Space, se houver conta PRO.

**Roteiro para demonstrar ao vivo** (inicie uns 10 minutos antes; o link só vale enquanto a célula roda):
1. "Por que existe o Anjo do Norte?" na aba **Comparar**: mostra o naive e o avançado.
2. "Quais foram os resultados do desenvolvimento ferroviário na Flórida?": no naive, a passagem certa estava na
   posição 15.
3. "Qual foi o placar da final da Copa do Mundo de 2026?": mostra a **recusa**.
4. Ligar o **HyDE** e abrir "Documento hipotético" para mostrar o texto que o LLM inventou.

---

## 15. Engenharia: organização, testes e reprodutibilidade

**Organização:**
- `src/rag_core.py` tem as **funções puras** (sem modelos): chunking, tokenização, RRF, métricas e prompts. São
  usadas pela interface e testadas em `tests/`.
- O notebook tem uma cópia dessas funções numa célula, para rodar no Colab sem depender do repositório.
- `space/app.py` importa as funções de `src/`, então não há código duplicado entre a interface e os testes.

**Testes** (`tests/test_rag_core.py`, 9 testes com `pytest`):
- chunking respeita o tamanho máximo, lida com texto vazio e com palavra gigante;
- tokenização remove acentos e *stopwords*;
- RRF premia quem aparece nas duas listas;
- conversão chunk → documento remove duplicados;
- Recall, MRR e nDCG dão os valores esperados em casos pequenos calculados à mão;
- o prompt numera os trechos.

**Por que testar funções puras?** Um erro nas métricas invalidaria todas as conclusões. Elas são testadas em casos
cujo resultado sabemos de cabeça.

**Reprodutibilidade:**
- Semente fixa (`SEED = 42`) para `random`, `numpy` e `torch`.
- Decodificação gulosa (`do_sample=False`).
- Todos os modelos e o dataset são públicos no Hub, com nomes fixos.
- Os resultados ficam salvos em `results/resultados_rag.json`.
- **Ressalva:** a latência varia entre execuções (depende da GPU alocada no Colab).

**Onde entra o ecossistema Hugging Face** (o enunciado pede):

| Componente | Uso no projeto |
|---|---|
| Hugging Face Datasets | Corpus, perguntas e gabarito (`load_dataset`); publicação do índice (`push_to_hub`) |
| Hub de modelos | bge-m3, bge-reranker-v2-m3, Qwen3-4B |
| `sentence-transformers` | Embeddings (`SentenceTransformer`) e reranking (`CrossEncoder`) |
| `transformers` + `bitsandbytes` | LLM em 4 bits |
| Inference Providers | Alternativa para chamar o LLM sem GPU (`USE_API` no notebook) |
| Gradio | Interface da demo |

---

## 16. Contexto histórico: GPTs de 2022 a 2026

O enunciado pede que o projeto se situe na evolução dos GPTs. A ideia central para falar:

| Fase | O que mudou | Efeito no RAG |
|---|---|---|
| **2022** | InstructGPT/RLHF, ChatGPT, HyDE | Modelos passam a **seguir instruções** ("use só o contexto"); surge o HyDE |
| **2023** | LLaMA, Mistral, QLoRA (4 bits), *Lost in the Middle* | Modelos abertos que rodam em GPU barata; RAG vira padrão; descobre-se que **ordem e quantidade** de trechos importam |
| **2024** | BGE-M3, rerankers multilíngues, contexto de 1M, GraphRAG | "Contexto longo vai substituir o RAG?" **Não**: é caro e o modelo se distrai |
| **2025** | DeepSeek-R1, Qwen3, *Context Rot*, RAG agêntico | Modelos de raciocínio; RAG vira um **agente** que decide quando buscar |
| **2026** | Consolidação | Agentes com protocolos abertos (MCP), recuperação multimodal, avaliação contínua |

**Argumento que conecta tudo ao projeto:** mesmo com janelas de contexto enormes, estudos como *Lost in the
Middle* (2023) e *Context Rot* (2025) mostram que o modelo responde melhor quando recebe **pouco texto e
relevante**. É exatamente o que o **reranking** entrega, e foi a técnica que mais ajudou no experimento. E mesmo um
agente moderno depende de uma boa função de busca: "um agente que busca mal só erra mais vezes".

---

## 17. Limitações e trabalhos futuros

**Limitações (dizer antes que perguntem mostra maturidade):**
- **50 perguntas** é pouco: diferenças menores que ~0,03 podem ser ruído (0,02 de Recall@5 = uma pergunta).
- Corpus traduzido automaticamente.
- Avaliamos a **recuperação**, não a **qualidade da resposta** gerada.
- Busca densa medida em lote (latência subestimada).
- Não rodamos a ablação **"denso + rerank"** (sem BM25), que diria se o BM25 contribui depois do reranking.
- Não testamos pesos diferentes no RRF nem tamanhos de chunk diferentes.
- O GraphRAG é uma demonstração com 12 trechos, sem avaliação.

**Trabalhos futuros:**
- Ablação denso + rerank e RRF ponderado.
- Avaliar a resposta com **RAGAS** (*faithfulness*: a resposta é fiel ao contexto? *answer relevancy*: responde à
  pergunta?).
- *Contextual retrieval*: antes de indexar, o LLM escreve uma frase situando cada chunk no documento.
- Comparar com a família Qwen3-Embedding/Reranker.
- Stemming no BM25.
- GraphRAG completo com resolução de entidades.
- RAG agêntico: o LLM decide quando e o que buscar, com a busca exposta via MCP.

---

## 18. Roteiro sugerido para a apresentação

Cerca de 12 a 15 minutos, mais perguntas.

| Tempo | Parte | O que dizer | O que mostrar |
|---|---|---|---|
| 1 min | Problema | LLM tem conhecimento estático, alucina e não cita fonte. RAG resolve. | — |
| 2 min | Contexto 2022–2026 | Contexto longo não substituiu RAG; selecionar pouco e bem continua decisivo. | Tabela da seção 16 |
| 1 min | Objetivo e dados | Medir com gabarito quanto cada técnica ajuda; NanoNQ-pt, 50 perguntas. | README |
| 3 min | Pipeline | Chunking → denso + BM25 → RRF → cross-encoder → LLM com citações; HyDE opcional. | Diagrama do README |
| 3 min | Resultados | Rerank foi o maior ganho; híbrido piorou; HyDE neutro e caro. Explicar o porquê de cada um. | Tabela + gráfico |
| 3 min | Demo ao vivo | Anjo do Norte, Flórida, Copa 2026 (recusa), HyDE ligado. | Gradio |
| 1 min | GraphRAG | Perguntas globais; o que o grafo mostrou e seus limites. | `graphrag_demo.png` |
| 1 min | Limitações e futuro | 50 perguntas, sem ablação denso + rerank, sem avaliação da resposta. | — |

**Dicas:**
- Comece a demo antes da apresentação e deixe a aba aberta.
- Tenha um **plano B**: se o Colab cair, mostre o notebook já executado no Drive.
- Quando não souber algo, diga o que faria para descobrir ("rodaria a ablação X").

---

## 19. Perguntas prováveis do professor (com respostas)

<details>
<summary><b>1. O que é RAG e por que não fazer fine-tuning?</b></summary>

RAG busca trechos relevantes numa base e os coloca no prompt, para o LLM responder apoiado neles e citando a fonte.
Fine-tuning muda os pesos do modelo: é caro, precisa ser refeito quando os dados mudam e não dá rastreabilidade.
RAG é melhor para conhecimento factual e que muda; fine-tuning, para estilo ou tarefa específica.
</details>

<details>
<summary><b>2. Por que vocês escolheram o bge-m3?</b></summary>

É aberto, multilíngue (forte em português), gera vetores de 1024 dimensões, aceita textos longos (8K tokens) e foi
treinado especificamente para busca, com pares pergunta–passagem. O enunciado pede `sentence-transformers`, e ele
funciona direto nessa biblioteca.
</details>

<details>
<summary><b>3. Por que normalizar os vetores?</b></summary>

Com vetores de comprimento 1, o produto interno é igual à similaridade de cosseno. Assim dá para usar o
`IndexFlatIP` do FAISS, que é simples e rápido, e o cosseno ignora o tamanho do vetor, comparando só a direção
(o significado).
</details>

<details>
<summary><b>4. Por que FAISS com busca exata? E se a base tivesse 10 milhões de documentos?</b></summary>

Com 6.643 vetores, a busca exata leva milissegundos, então não há motivo para aproximar. Com milhões, usaríamos um
índice aproximado (HNSW ou IVF, com quantização dos vetores), ou um banco vetorial (Chroma, Qdrant, Milvus). O
cross-encoder continuaria só nos top-30, então o custo do rerank não cresce com o tamanho da base.
</details>

<details>
<summary><b>5. Como escolheram 800 caracteres e 120 de sobreposição?</b></summary>

São valores intermediários comuns: o trecho cabe um fato com contexto, e 5 trechos cabem no prompt. A sobreposição
evita perder informação que cai na fronteira entre dois trechos. Não otimizamos esses valores; seria um trabalho
futuro, medindo o Recall com tamanhos diferentes.
</details>

<details>
<summary><b>6. Por que a busca híbrida piorou, se a literatura diz que ela ajuda?</b></summary>

O RRF dá peso igual às duas listas, e o BM25 foi muito mais fraco aqui (Recall@5 de 0,43 contra 0,64). O 1º lugar
do BM25, muitas vezes errado, recebe o mesmo score do 1º lugar do denso e empurra os acertos para baixo. O BM25 foi
fraco porque as perguntas são paráfrases em linguagem natural, o corpus é traduzido automaticamente e não usamos
stemming. Busca híbrida ajuda quando os dois buscadores têm qualidade parecida e erram em perguntas diferentes.
Para corrigir: RRF ponderado ou testar "denso + rerank" sem BM25.
</details>

<details>
<summary><b>7. Por que RRF e não somar os scores? O que é o 60?</b></summary>

Os scores do BM25 e do cosseno estão em escalas diferentes e não comparáveis. O RRF usa só a posição:
`1/(60 + posição)`. O 60 suaviza a diferença entre posições, para que o 1º lugar de uma lista não domine. É o valor
do artigo original (Cormack et al., 2009) e o padrão da área.
</details>

<details>
<summary><b>8. Qual a diferença entre bi-encoder e cross-encoder?</b></summary>

O bi-encoder codifica pergunta e documento separados em vetores; dá para pré-calcular os documentos, então é rápido.
O cross-encoder lê pergunta e trecho juntos, com atenção entre todos os tokens dos dois, e dá um score de relevância.
É bem mais preciso, mas precisa rodar para cada par, então só cabe para poucos candidatos.
</details>

<details>
<summary><b>9. Por que 30 candidatos no reranker e 5 no contexto?</b></summary>

30 é um equilíbrio: cobertura suficiente (a passagem certa geralmente está entre eles) e custo baixo (~230 ms).
5 trechos no contexto porque *Lost in the Middle* e *Context Rot* mostram que mais texto não significa resposta
melhor: o modelo se distrai. Por isso também pegamos um trecho por documento, para diversificar.
</details>

<details>
<summary><b>10. O que é o HyDE e por que não funcionou?</b></summary>

O LLM escreve uma resposta hipotética e buscamos com o embedding dela, que se parece mais com um documento do que a
pergunta. Não funcionou porque: (1) o HyDE foi pensado para buscadores sem treino supervisionado, e o bge-m3 já foi
treinado com pares pergunta–passagem; (2) o LLM de 4B às vezes inventa fatos errados; (3) o reranker reordena com a
pergunta original e neutraliza o efeito. E custa ~9 s por pergunta.
</details>

<details>
<summary><b>11. Se o texto do HyDE pode estar errado, isso não gera alucinação?</b></summary>

Não, porque ele é usado **só para a busca densa**. O BM25 e o reranker usam a pergunta original, e o texto
hipotético nunca entra no prompt final. A resposta é gerada só a partir dos trechos reais recuperados.
</details>

<details>
<summary><b>12. O que é quantização em 4 bits? Não perde qualidade?</b></summary>

É guardar cada peso com 4 bits em vez de 16, o que reduz o modelo de ~8 GB para ~3 GB e permite rodar numa T4
gratuita. Usamos o NF4 (QLoRA), que distribui os níveis seguindo uma curva normal, como os pesos costumam se
distribuir, e perde pouca informação. Há uma pequena perda de qualidade, aceitável para o objetivo.
</details>

<details>
<summary><b>13. Por que o Qwen3-4B?</b></summary>

É aberto (Apache 2.0), recente (julho/2025), bom em português, segue bem instruções como "cite as fontes" e cabe
numa GPU gratuita em 4 bits. Um modelo maior responderia melhor, mas não rodaria de graça.
</details>

<details>
<summary><b>14. Como vocês reduzem alucinação? Funciona sempre?</b></summary>

O prompt manda usar **só** o contexto, citar `[n]` e responder uma frase fixa de recusa quando não houver resposta.
No teste com a Copa de 2026 (fora da base), o modelo recusou. Mas não é garantido: na pergunta do Anjo do Norte, com
a passagem certa fora do contexto, o modelo respondeu com base em trechos de outro assunto. Por isso a recuperação
boa é tão importante: o LLM só é tão bom quanto o contexto que recebe.
</details>

<details>
<summary><b>15. Explique Recall@5, MRR@10 e nDCG@10.</b></summary>

Recall@5: fração das passagens corretas que estão no top-5 (se a resposta chega ao LLM). MRR@10: 1 dividido pela
posição da primeira correta (se ela está no topo). nDCG@10: soma os acertos com desconto logarítmico pela posição,
normalizado pelo ideal. Recall mede cobertura; MRR e nDCG medem ordem.
</details>

<details>
<summary><b>16. Por que avaliar por documento e não por chunk?</b></summary>

O gabarito do dataset marca documentos relevantes, não chunks. Convertemos o ranking de chunks em ranking de
documentos mantendo a primeira aparição de cada um (`to_doc_ranking`).
</details>

<details>
<summary><b>17. 50 perguntas é suficiente para concluir algo?</b></summary>

Para diferenças grandes, sim: o ganho do reranker em MRR (0,444 → 0,634) é bem maior que o ruído. Para diferenças
pequenas, não: o +0,02 do HyDE equivale a uma pergunta e pode ser acaso. Por isso dizemos que o HyDE foi "neutro",
não que "melhorou". Para afirmar com mais segurança, usaríamos mais perguntas e um teste estatístico (por exemplo,
teste pareado ou *bootstrap*).
</details>

<details>
<summary><b>18. Por que a latência do HyDE não está na tabela?</b></summary>

A coluna mede só a recuperação, para comparar as configurações de forma justa. O HyDE gasta ~9,4 s por pergunta
gerando o texto (7 min 48 s para 50), o que deixamos explícito no README e no relatório. Também a busca densa foi
medida em lote, então os 3 ms estão um pouco subestimados.
</details>

<details>
<summary><b>19. O que é GraphRAG e quando usar?</b></summary>

É extrair entidades e relações com o LLM, montar um grafo, agrupar em comunidades e resumir cada uma. Serve para
perguntas **globais** ("quais os temas principais?"), que o RAG por chunks não responde porque a resposta não está
num trecho só. É caro (muitas chamadas ao LLM). Nossa versão é uma demonstração com 12 trechos; mostrou comunidades
coerentes, mas também entidades duplicadas e vagas, porque falta resolução de entidades.
</details>

<details>
<summary><b>20. Onde o Hugging Face entra no projeto?</b></summary>

Dataset do Hub (`load_dataset`), os três modelos do Hub, `sentence-transformers` para embeddings e reranking,
`transformers` + `bitsandbytes` para o LLM, publicação do índice como dataset (`push_to_hub`), Inference Providers
como alternativa sem GPU e Gradio para a interface.
</details>

<details>
<summary><b>21. Por que a demo não está num Space?</b></summary>

O Hugging Face passou a exigir conta PRO para Spaces com Gradio, e os créditos gratuitos do Inference Providers
acabaram (erro 402). Rodamos no Colab com GPU T4 e o LLM local, com link público do Gradio. O app continua compatível
com um Space.
</details>

<details>
<summary><b>22. Se tivessem mais tempo, o que fariam primeiro?</b></summary>

A ablação "denso + rerank", para saber se o BM25 contribui depois do reranking. Depois, RRF ponderado e avaliação da
resposta gerada com RAGAS. São os passos que mais responderiam às dúvidas que os resultados deixaram.
</details>

<details>
<summary><b>23. Contexto longo (1M de tokens) não torna o RAG desnecessário?</b></summary>

Não. Colocar a base inteira no prompt é caro, lento e o modelo se distrai: *Lost in the Middle* mostrou que ele
ignora informação no meio de contextos longos, e *Context Rot* que o desempenho cai com o tamanho do contexto. Além
disso, bases grandes não cabem e mudam o tempo todo. Selecionar pouco e bem continua decisivo.
</details>

<details>
<summary><b>24. Como garantem que as métricas estão certas?</b></summary>

As funções de métrica estão em `src/rag_core.py` e têm testes unitários com casos calculados à mão (por exemplo,
MRR de uma lista com o acerto na posição 2 tem que dar 0,5). Também usamos as definições padrão do BEIR.
</details>

<details>
<summary><b>25. Como reproduzir o experimento?</b></summary>

Abrir o notebook no Colab com T4 e executar tudo (~20–30 min). Semente fixa, decodificação gulosa, modelos e dataset
públicos com nomes fixos. As métricas de recuperação se repetem; a latência varia com a GPU.
</details>

---

## 20. Glossário

| Termo | Significado |
|---|---|
| **Ablação** | Experimento que remove uma peça para medir quanto ela contribui. |
| **BM25** | Algoritmo de busca por palavras exatas, com saturação de frequência, peso para palavras raras e normalização por tamanho. |
| **Bi-encoder** | Modelo que codifica pergunta e documento separadamente em vetores. |
| **Chunk** | Pedaço de um documento usado como unidade de busca. |
| **Cross-encoder** | Modelo que lê pergunta e trecho juntos e devolve um score de relevância. |
| **Embedding** | Vetor que representa o significado de um texto. |
| **FAISS** | Biblioteca de busca vetorial da Meta. |
| **GraphRAG** | RAG sobre um grafo de entidades e comunidades, para perguntas globais. |
| **HyDE** | Busca com o embedding de uma resposta hipotética gerada pelo LLM. |
| **Louvain** | Algoritmo de detecção de comunidades que maximiza a modularidade do grafo. |
| **MRR** | Média de 1/posição da primeira resposta correta. |
| **NF4** | Formato de quantização em 4 bits que segue uma distribuição normal (QLoRA). |
| **nDCG** | Métrica de ranking com desconto logarítmico pela posição, normalizada pelo ideal. |
| **Qrels** | Gabarito: pares (pergunta, documento relevante). |
| **Recall@k** | Fração das respostas corretas que aparecem entre os k primeiros resultados. |
| **RRF** | Fusão de rankings pela soma de 1/(k + posição). |
| **Stemming** | Redução das palavras ao radical ("ferroviário" → "ferrovi"). |
| **Stopwords** | Palavras muito comuns e pouco informativas ("de", "a", "que"). |

---

## 21. Cola: números para saber de cor

| O quê | Valor |
|---|---|
| Documentos / chunks / perguntas | 5.035 / 6.643 / 50 |
| Chunk / sobreposição | 800 / 120 caracteres |
| Candidatos por buscador / no rerank / no contexto | 50 / 30 / 5 |
| RRF k | 60 |
| Dimensão do embedding | 1024 |
| Recall@5: denso → híbrido → rerank → HyDE | 0,64 → 0,61 → 0,70 → 0,72 |
| MRR@10: denso → híbrido → rerank | 0,548 → 0,444 → **0,634** |
| Recall@1: denso → rerank | 0,39 → 0,49 |
| Perguntas com a resposta fora do top-5 no naive | 15 de 50 (o rerank resgatou 5) |
| Latência de recuperação: híbrido → rerank | ~11 ms → ~235 ms |
| Custo do HyDE | ~9,4 s por pergunta (7 min 48 s para 50) |
| LLM | Qwen3-4B-Instruct-2507, 4 bits NF4, ~3 GB de VRAM |
| GPU | Colab T4 (16 GB) |
