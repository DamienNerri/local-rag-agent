import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client

# 1. Configuration des modèles locaux via Ollama
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_URL, request_timeout=600.0, embed_batch_size=1)

DOCUMENTS_DIR = "./documents"
COLLECTION_NAME = "local_rag"

# 2. Connexion à Qdrant
def get_qdrant_client():
    qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
    host = qdrant_host.replace("http://", "").split(":")[0]
    port = int(qdrant_host.split(":")[-1])
    return qdrant_client.QdrantClient(host=host, port=port, timeout=600)

# 3. Ingestion automatique si la collection n'existe pas
def ensure_index():
    client = get_qdrant_client()
    collection_exists = client.collection_exists(COLLECTION_NAME)

    if not collection_exists:
        print("📂 [Ingestion] Collection absente, indexation des documents...")
        if not os.path.exists(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
            print("⚠️  Aucun document trouvé dans ./documents/")
            return None
        from pdf2image import convert_from_path
        import pytesseract
        import glob

        documents = []
        for pdf_path in glob.glob(os.path.join(DOCUMENTS_DIR, "**/*.pdf"), recursive=True):
            print(f"  📄 OCR sur {pdf_path}...")
            images = convert_from_path(pdf_path)
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img, lang="fra")
                if text.strip():
                    documents.append(Document(text=text, metadata={"source": pdf_path, "page": i + 1}))

        # Aussi charger les fichiers texte classiques
        for ext in ("*.txt", "*.md"):
            for fpath in glob.glob(os.path.join(DOCUMENTS_DIR, "**/" + ext), recursive=True):
                with open(fpath) as f:
                    documents.append(Document(text=f.read(), metadata={"source": fpath}))

        if not documents:
            print("⚠️  Aucun contenu extrait des documents.")
            return None

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents)
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        print(f"✅ [Ingestion] {len(nodes)} chunks indexés depuis {len(documents)} pages.")
    else:
        print("✅ [Ingestion] Collection existante, chargement...")
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    return index

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions en te basant UNIQUEMENT "
    "sur les documents fournis ci-dessous. Réponds en français."
)
TOP_K = 3


def build_query_engine():
    """Construit le moteur de requête (indexe les documents au besoin)."""
    index = ensure_index()
    if not index:
        return None
    return index.as_query_engine(similarity_top_k=TOP_K, system_prompt=SYSTEM_PROMPT)


def format_sources(response):
    """Liste dédoublonnée « fichier, p. N (score) » des chunks utilisés pour répondre."""
    lines, seen = [], set()
    for sn in response.source_nodes:
        meta = sn.node.metadata
        source = os.path.basename(meta.get("source", "inconnu"))
        page = meta.get("page")
        label = f"{source}, p. {page}" if page else source
        if label in seen:
            continue
        seen.add(label)
        score = f" (score {sn.score:.2f})" if sn.score is not None else ""
        lines.append(f"  - {label}{score}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("🤖 [OmniRAG] Initialisation du RAG local...")
    try:
        query_engine = build_query_engine()
        if not query_engine:
            print("❌ Impossible de démarrer sans documents.")
            exit(1)
        print("✅ [OmniRAG] RAG prêt. Posez vos questions sur vos documents.")

        while True:
            try:
                query = input("\n👤 Vous : ")
            except EOFError:
                break
            if not query or query.lower() in ["exit", "quit"]:
                if not query:
                    import time; time.sleep(1)
                    continue
                break

            print("🧠 Recherche dans vos documents...")
            response = query_engine.query(query)
            print(f"\n🤖 Assistant :\n{response}")
            sources = format_sources(response)
            if sources:
                print(f"\n📚 Sources :\n{sources}")

    except Exception as e:
        print(f"❌ Erreur au démarrage : {e}")
        import traceback; traceback.print_exc()
        print("Assurez-vous que le conteneur Qdrant et Ollama tournent correctement.")
