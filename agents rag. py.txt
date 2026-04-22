# src/cloud_agent/graph.py
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[Sequence[str], "Historial de auditoría"]
    infrastructure_data: dict
    risk_level: str

class CloudArchitectAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
    def security_audit(self, state: AgentState):
        # Lógica simulada de escaneo de S3 buckets abiertos o IAM débil
        data = state['infrastructure_data']
        print("--- Nodo: Auditoría de Seguridad ---")
        return {"risk_level": "High" if data.get('open_ports') else "Low"}

    def cost_optimization(self, state: AgentState):
        # Lógica para detectar instancias EC2 subutilizadas
        print("--- Nodo: Optimización de Costos ---")
        return {"messages": ["Sugerencia: Migrar instancias t2.micro a t3.micro para ahorrar 15%"]}

# Construcción del Grafo de Decisión
workflow = StateGraph(AgentState)
agent = CloudArchitectAgent()

workflow.add_node("security", agent.security_audit)
workflow.add_node("costs", agent.cost_optimization)

workflow.set_entry_point("security")
workflow.add_edge("security", "costs")
workflow.add_edge("costs", END)

app = workflow.compile()