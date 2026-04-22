import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Configuración Inicial
load_dotenv()

class PortfolioRAG:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.vector_db = self._setup_vector_db()

    def _setup_vector_db(self):
        """Carga documentos, los divide en fragmentos y los indexa."""
        print("--- Indexando Proyectos del Portafolio ---")
        loader = DirectoryLoader(self.docs_path, glob="./*.md", loader_cls=TextLoader)
        docs = loader.load()
        
        # Dividir el código/texto en fragmentos inteligentes para no perder contexto
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Crear base de datos vectorial en memoria (puedes persistirla en disco)
        return Chroma.from_documents(documents=splits, embedding=self.embeddings)

    def get_agent_chain(self):
        """Crea una cadena de RAG con memoria de conversación."""
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        # Contextualizar la pregunta del usuario según el historial
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Dada una conversación y la última pregunta del usuario, formula una pregunta independiente."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, context_prompt)

        # Prompt del sistema para el comportamiento del agente
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres el asistente técnico de Elvis Soto. Responde preguntas sobre sus proyectos usando el contexto: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 3. Ejecución del Agente
if __name__ == "__main__":
    # Asegúrate de tener una carpeta 'proyectos' con archivos .md
    if not os.path.exists("./proyectos"):
        os.makedirs("./proyectos")
        with open("./proyectos/ejemplo.md", "w") as f:
            f.write("# Proyecto Alpha\nStack: Ruby on Rails, AWS.\nDescripción: Sistema de microservicios para logística.")

    rag_system = PortfolioRAG(docs_path="./proyectos")
    agent = rag_system.get_agent_chain()
    
    chat_history = []
    print("\n--- Agente de Portafolio Activo (Escribe 'salir' para terminar) ---")
    
    while True:
        user_input = input("Tú: ")
        if user_input.lower() == 'salir': break
        
        response = agent.invoke({"input": user_input, "chat_history": chat_history})
        print(f"\nAgente: {response['answer']}\n")
        
        # Actualizar historial para que el agente tenga memoria
        chat_history.extend([("human", user_input), ("ai", response["answer"])])