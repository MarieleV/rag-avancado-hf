# Documentação

| Arquivo | Conteúdo |
|---|---|
| [`relatorio.md`](relatorio.md) | Relatório de pesquisa: evolução dos GPTs (2022–2026), estado da arte em RAG, metodologia, resultados e referências |
| [`apresentacao.md`](apresentacao.md) | Slides da apresentação (formato [Marp](https://marp.app/)) |

## Gerar os PDFs

- **Relatório:** abra `relatorio.md` no VS Code com a extensão *Markdown PDF* e use **Markdown PDF: Export (pdf)**,
  ou rode `pandoc relatorio.md -o relatorio.pdf`.
- **Slides:** instale a extensão *Marp for VS Code*, abra `apresentacao.md` e use **Marp: Export Slide Deck → PDF**,
  ou rode `npx @marp-team/marp-cli apresentacao.md --pdf --allow-local-files`.

As seções marcadas com ✏️ devem ser preenchidas com os números gerados pelo notebook (seção 10).
