"""Évaluation du RAG avec Ragas sur un jeu de questions de référence.

Corpus : eval/corpus/ (documents fictifs, dont un PDF scanné pour tester l'OCR),
indexé dans une collection Qdrant dédiée, recréée à chaque lancement.

Usage (depuis la racine du repo, avec Qdrant et Ollama lancés) :
    pip install -r requirements-eval.txt
    python eval/run_ragas.py

Le juge est un modèle Ollama local (EVAL_JUDGE_MODEL, qwen2.5:7b par défaut) :
un modèle de 3B est trop faible pour juger de façon fiable.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Configuration à définir AVANT d'importer main.py (lue au chargement du module)
os.environ.setdefault("DOCUMENTS_DIR", "./eval/corpus")
os.environ.setdefault("COLLECTION_NAME", "eval_corpus")
os.environ.setdefault("QDRANT_HOST", "http://localhost:6333")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from main import COLLECTION_NAME, LLM_MODEL, OLLAMA_URL, build_query_engine, get_qdrant_client

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "qwen2.5:7b")
QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def collect_samples(query_engine, questions):
    samples = []
    for i, item in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {item['question']}")
        response = query_engine.query(item["question"])
        samples.append({
            "user_input": item["question"],
            "response": str(response),
            "retrieved_contexts": [sn.node.get_content() for sn in response.source_nodes],
            "reference": item["reference"],
        })
    return samples


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if any("À REMPLACER" in q["question"] for q in questions):
        sys.exit("❌ Remplissez d'abord eval/questions.json avec vos propres questions/réponses.")

    client = get_qdrant_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)  # réindexation propre à chaque run
    query_engine = build_query_engine()
    if not query_engine:
        sys.exit("❌ Index vide : ajoutez des documents dans ./documents.")

    print(f"🧪 Génération des réponses pour {len(questions)} questions...")
    dataset = EvaluationDataset.from_list(collect_samples(query_engine, questions))

    judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL))

    print(f"⚖️  Évaluation avec le juge {JUDGE_MODEL}...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),                      # la réponse est-elle fondée sur les contextes ?
            ResponseRelevancy(),                 # répond-elle à la question ?
            LLMContextPrecisionWithReference(),  # les chunks remontés sont-ils pertinents ?
            LLMContextRecall(),                  # contiennent-ils l'info de la référence ?
        ],
        llm=judge,
        embeddings=embeddings,
        # Ollama local traite les requêtes en série : peu de parallélisme, timeout large
        run_config=RunConfig(max_workers=2, timeout=600),
    )
    print("\n📊 Scores moyens :")
    print(result)

    df = result.to_pandas()
    out = Path(__file__).parent / f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out, index=False)
    print(f"💾 Détail par question : {out}")

    metrics = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
    report = Path(__file__).parent / "RESULTS.md"
    lines = [
        "# Résultats de l'évaluation Ragas", "",
        f"- Date : {datetime.now():%Y-%m-%d}",
        f"- Modèle RAG : `{LLM_MODEL}`, juge : `{JUDGE_MODEL}`, {len(df)} questions", "",
        "## Moyennes", "", "| Métrique | Score |", "|---|---|",
        *[f"| {m} | {df[m].mean():.2f} |" for m in metrics], "",
        "## Détail par question", "",
        "| Question | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
        *["| " + r["user_input"] + " | " + " | ".join(f"{r[m]:.2f}" for m in metrics) + " |" for _, r in df.iterrows()],
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Synthèse : {report}")


if __name__ == "__main__":
    main()
