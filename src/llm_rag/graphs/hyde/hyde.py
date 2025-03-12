from typing import TypedDict

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

from llm_rag import embeddings, llm, vectorstore

hyde_prompt_template = """Please write a passage to answer the question
Question: {question}
Passage:"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)


rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class State(TypedDict):
    question: str
    generated_documents: list[str]
    hyde_embeddings: np.ndarray
    context: list[Document]
    answer: str


def generate_documents(state: State, config: RunnableConfig) -> list[Document]:
    generated_documents_count = config["configurable"].get(
        "generated_documents_count", 3
    )

    chain = hyde_prompt | llm | StrOutputParser()
    generated_documents = chain.batch(
        [{"question": state["question"]}] * generated_documents_count
    )

    return {"generated_documents": generated_documents}


def calculate_hyde_embeddings(state: State):
    question_embeddings = np.array(embeddings.embed_query(state["question"]))
    generated_documents_embeddings = np.array(
        embeddings.embed_documents(state["generated_documents"])
    )
    hyde_embeddings = np.vstack(
        [question_embeddings, generated_documents_embeddings]
    ).mean(axis=0)
    return {"hyde_embeddings": hyde_embeddings}


def get_relevant_documents(state: State):
    documents = vectorstore.similarity_search_by_vector(state["hyde_embeddings"])
    return {"context": documents}


def generate_answer(state: State):
    docs_content = format_docs(state["context"])
    chain = rag_prompt | llm | StrOutputParser()
    response = chain.invoke({"question": state["question"], "context": docs_content})
    return {"answer": response}


class ConfigSchema(BaseModel):
    generated_documents_count: int = Field(default=3, gt=0)


graph_builder = StateGraph(State, ConfigSchema)

graph_builder.add_node("generate_documents", generate_documents)
graph_builder.add_node("calculate_hyde_embeddings", calculate_hyde_embeddings)
graph_builder.add_node("get_relevant_documents", get_relevant_documents)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "generate_documents")
graph_builder.add_edge("generate_documents", "calculate_hyde_embeddings")
graph_builder.add_edge("calculate_hyde_embeddings", "get_relevant_documents")
graph_builder.add_edge("get_relevant_documents", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    query = "What is task decomposition for LLM agents?"
    config = {
        "configurable": {
            "generated_documents_count": 5,
        }
    }
    response = graph.invoke(
        {"question": query},
        config=config,
    )

    rprint(Pretty(response, max_depth=2))
    rprint(Markdown(response["answer"]))
