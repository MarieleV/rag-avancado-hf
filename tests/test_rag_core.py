import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_core import (  # noqa: E402
    build_messages, chunk_text, evaluate, mrr_at_k, ndcg_at_k,
    recall_at_k, rrf_fuse, to_doc_ranking, tokenize,
)


def test_chunk_respeita_tamanho_maximo():
    texto = "A capital do Brasil é Brasília. " * 50
    chunks = chunk_text(texto, size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_texto_curto_e_vazio():
    assert chunk_text("Frase curta.") == ["Frase curta."]
    assert chunk_text("") == []


def test_chunk_palavra_gigante_sem_espacos():
    chunks = chunk_text("x" * 1000, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)


def test_tokenize_remove_acentos_e_stopwords():
    assert tokenize("Qual é a população de São Paulo?") == ["populacao", "sao", "paulo"]


def test_rrf_premia_quem_aparece_nas_duas_listas():
    fused = rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]


def test_to_doc_ranking_remove_duplicados():
    assert to_doc_ranking([0, 1, 2, 3], ["d1", "d1", "d2", "d3"]) == ["d1", "d2", "d3"]


def test_metricas():
    assert recall_at_k(["a", "b"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert mrr_at_k(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], {"a"}) == 1.0


def test_evaluate_media():
    res = evaluate({"q1": ["d1"], "q2": ["x"]}, {"q1": {"d1"}, "q2": {"d2"}})
    assert res["Recall@1"] == 0.5
    assert res["MRR@10"] == 0.5


def test_prompt_numera_os_trechos():
    msgs = build_messages("Pergunta?", ["trecho A", "trecho B"])
    assert msgs[0]["role"] == "system"
    assert "[1] trecho A" in msgs[1]["content"] and "[2] trecho B" in msgs[1]["content"]
