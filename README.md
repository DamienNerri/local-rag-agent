# Local RAG

Un RAG (Retrieval-Augmented Generation) 100 % local pour interroger ses propres documents
en français, y compris des PDF scannés, sans qu'aucune donnée ne quitte la machine.

> **Statut :** prototype en ligne de commande. C'est un RAG, pas (encore) un agent :
> le modèle ne choisit pas d'outils, il répond à partir des passages retrouvés.

## Fonctionnement

```
documents/*.pdf ──OCR (Tesseract fra)──┐
documents/*.txt, *.md ─────────────────┤
                                       ▼
                     découpage (SentenceSplitter 1024 / 100)
                                       ▼
                 embeddings nomic-embed-text (Ollama) ──► Qdrant
                                                           │
question ──► embedding ──► top-3 chunks ──────────────────┘
                                       ▼
                     qwen2.5:3b (Ollama) ──► réponse + sources (fichier, page)
```

- **Ingestion** : au premier lancement, si la collection Qdrant n'existe pas, les PDF sont
  passés à l'OCR page par page (les métadonnées `source` et `page` sont conservées),
  découpés puis indexés.
- **Réponse** : les 3 chunks les plus proches sont injectés dans le prompt, avec la consigne
  de ne répondre qu'à partir des documents. Les sources utilisées sont affichées sous
  chaque réponse.

## Stack

| Brique | Choix |
|---|---|
| Orchestration | LlamaIndex |
| LLM | `qwen2.5:3b` via Ollama (surchargeable avec `LLM_MODEL`) |
| Embeddings | `nomic-embed-text` via Ollama |
| Base vectorielle | Qdrant |
| OCR | Tesseract (`fra`) + pdf2image |
| Évaluation | Ragas |
| Conteneurisation | Docker Compose |

## Lancer le projet

Prérequis : [Ollama](https://ollama.com) installé sur la machine hôte, et Docker.

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

mkdir -p documents          # déposez-y vos PDF / .txt / .md (ignoré par git)
docker compose up -d qdrant
docker compose run --rm rag
```

Pour réindexer après avoir changé les documents, supprimez la collection :
`curl -X DELETE http://localhost:6333/collections/local_rag`.

## Évaluation

Le RAG est évalué avec Ragas sur un corpus **fictif** (`eval/corpus/`) :
un guide RH en Markdown et une procédure d'astreinte en **PDF scanné**, pour tester aussi l'OCR.
Le jeu de référence (`eval/questions.json`) contient 12 questions : des faits simples, des faits
qui n'existent que dans le scan, une question qui croise les deux documents, et une question
hors corpus pour vérifier que le modèle n'invente pas de réponse.

```bash
pip install -r requirements-eval.txt     # Python 3.11
docker compose up -d qdrant
ollama pull qwen2.5:7b                   # modèle juge
python eval/run_ragas.py
```

Métriques : *faithfulness* (réponse fondée sur les passages), *answer relevancy*,
*context precision* et *context recall* (qualité du retrieval).
Derniers résultats : [`eval/RESULTS.md`](eval/RESULTS.md).

## Choix et limites assumés

- **Modèle de 3B paramètres** : choisi pour tourner sur un PC sans gros GPU. En production,
  je comparerais plusieurs tailles de modèle sur le jeu d'évaluation, selon le compromis
  qualité / latence / coût.
- **Ingestion non incrémentale** : l'index est construit une fois. En production, il faudrait
  une ingestion incrémentale avec un identifiant stable par document (hash du contenu) pour
  mettre à jour sans créer de doublons.
- **Interface CLI uniquement** : pas encore d'API.

## Roadmap

- [ ] API FastAPI (`/ingest`, `/query`) avec réponses sourcées
- [ ] Ingestion incrémentale (identifiant par document, suppression / mise à jour)
- [ ] Étoffer le jeu d'évaluation et le lancer en CI
- [ ] Comparaison de modèles et de paramètres de chunking sur les métriques Ragas
- [ ] Traçage des requêtes (latence, chunks retrouvés, scores)

## Contact

[LinkedIn](https://www.linkedin.com/in/damien-nerriere-605ab7221)
