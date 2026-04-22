import os
import sys
from typing import List

# Intentar importar librerías, si faltan, dar instrucciones claras
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: Faltan librerías. Ejecuta: pip install langchain langchain-openai chromadb python-dotenv")
    sys.exit(1)

# Cargar variables de entorno (API Key)
load_dotenv()

class PortfolioAI:
    def __init__(self, projects_data: List[dict]):
        """
        Inicializa el Agente con datos de proyectos dinámicos.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en el archivo .env")

        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = self._ingest_data(projects_data)
        self.chat_history = []

    def _ingest_data(self, data: List[dict]):
        """
        Transforma diccionarios de proyectos en una base de datos vectorial.
        """
        docs = []
        for p in data:
            content = f"Proyecto: {p['titulo']}\nStack: {p['stack']}\nDescripción: {p['descripcion']}\nReto: {p['reto']}"
            docs.append(Document(page_content=content, metadata={"source": p['titulo']}))

        # Splitter robusto para mantener contexto
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_docs = splitter.split_documents(docs)
        
        return Chroma.from_documents(final_docs, self.embeddings)

    def chat(self, question: str):
        """
        Maneja la lógica de RAG conversacional.
        """
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )
        
        result = qa.invoke({"question": question, "chat_history": self.chat_history})
        
        # Actualizar historial
        self.chat_history.append((question, result["answer"]))
        return result

# --- CONFIGURACIÓN DE TUS PROYECTOS ---
mis_proyectos = [
    {
        "titulo": "Enterprise Agentic-RAG Pipeline",
        "stack": "Python, LangChain, AWS Bedrock, Pinecone",
        "descripcion": "Sistema de agentes autónomos para análisis de documentos legales con 98% de precisión.",
        "reto": "Optimizar la latencia de recuperación de 5s a 200ms usando Hybrid Search."
    },
    {
        "titulo": "High-Performance Finance Engine",
        "stack": "Ruby on Rails, Redis, Sidekiq, PostgreSQL",
        "descripcion": "Backend distribuido para procesamiento de transacciones masivas en tiempo real.",
        "reto": "Implementar el Outbox Pattern para garantizar consistencia eventual en sistemas distribuidos."
    },
    {
        "titulo": "CUDA Video Analytics",
        "stack": "C++, OpenCV, NVIDIA TensorRT",
        "descripcion": "Motor de procesamiento de video de baja latencia para detección de objetos.",
        "reto": "Gestión manual de memoria en GPU para evitar fugas en streams de 24/7."
    }
]

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    print("🚀 Inicializando Agente de Portafolio Senior...")
    try:
        agent = PortfolioAI(mis_proyectos)
        print("✅ Agente listo. Pregúntame sobre mis proyectos (o escribe 'salir').\n")
        
        while True:
            user_query = input("Pregunta: ")
            if user_query.lower() in ["salir", "exit"]: break
            
            res = agent.chat(user_query)
            print(f"\n🤖 Agente: {res['answer']}")
            print(f"📄 Fuentes: {[d.metadata['source'] for d in res['source_documents']]}\n")
            
    except Exception as e:
        print(f"❌ Error crítico: {e}")