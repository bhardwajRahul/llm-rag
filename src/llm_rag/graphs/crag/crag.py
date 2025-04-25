import operator
from typing import Annotated, Literal, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

from llm_rag import llm
from llm_rag.indexing.reflection import retriever

rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

grading_prompt_template = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score to indicate whether the document is relevant to the question.

Retrieved document:
{document}

User question:
{question}"""

query_rewriting_prompt_template = """You a question re-writer that converts an input question to a better version that is optimized 
for web search. 
Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:
{question}

Formulate an improved question."""


def format_docs(docs: list[Document]) -> list[str]:
    return "\n\n".join(doc.page_content for doc in docs)


class DocumentGrade(BaseModel):
    """Relevance check on retrieved document."""

    chain_of_thought: str = Field(
        ...,
        description="Step by step reasoning to check if the document is relevant to the question",
    )
    is_relevant: bool = Field(description="Document is relevant to the question")


grader_llm = llm.with_structured_output(DocumentGrade, method="function_calling")


@chain
def grade_document(document, question):
    grading_prompt = grading_prompt_template.format(
        document=document, question=question
    )
    response = grader_llm.invoke([HumanMessage(content=grading_prompt)])
    return response


class WebSearchQuery(BaseModel):
    """Question optimization for web search."""

    chain_of_thought: str = Field(
        ..., description="Step by step reasoning to optimize query for web search"
    )
    web_search_query: str = Field(description="Optimized web search query")


web_search_llm = llm.with_structured_output(WebSearchQuery, method="function_calling")
web_search_tool = TavilySearchResults(k=4)


class State(TypedDict):
    question: str
    documents: list[Document]
    grades: list[DocumentGrade]
    is_web_search_required: bool
    web_search_query: str
    context: Annotated[list[Document], operator.add]
    answer: str


def retrieve(state: State):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}


def grade_documents(state: State):
    question = state["question"]
    documents = state["documents"]

    grades = grade_document.batch(documents, question=question)
    filtered_documents = [
        document for (document, grade) in zip(documents, grades) if grade.is_relevant
    ]
    is_web_search_required = len(filtered_documents) < len(documents)

    return {
        "context": filtered_documents,
        "grades": grades,
        "is_web_search_required": is_web_search_required,
    }


def check_documents_relevance(
    state: State,
) -> Literal["rewrite_query", "generate_answer"]:
    is_web_search_required = state["is_web_search_required"]

    if is_web_search_required:
        return "rewrite_query"
    else:
        return "generate_answer"


def rewrite_query(state: State):
    question = state["question"]
    query_rewriting_prompt = query_rewriting_prompt_template.format(question=question)
    response = web_search_llm.invoke(query_rewriting_prompt)
    return {"web_search_query": response.web_search_query}


def web_search(state: State):
    query = state["web_search_query"]
    results = web_search_tool.invoke({"query": query})
    documents = [Document(page_content=result["content"]) for result in results]
    return {"context": documents}


def generate_answer(state: State):
    docs_content = format_docs(state["context"])
    rag_prompt = rag_prompt_template.format(
        question=state["question"], context=docs_content
    )
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"answer": response.content}


graph_builder = StateGraph(State)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node("web_search", web_search)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_conditional_edges("grade_documents", check_documents_relevance)
graph_builder.add_edge("rewrite_query", "web_search")
graph_builder.add_edge("web_search", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    queries = [
        "What are common types of agent memory?",
        "What are main steps for collecting human data?",
        "How does the AlphaCodium paper work?",
    ]

    for query in queries:
        response = graph.invoke({"question": query})
        rprint(Pretty(response))
        rprint(Markdown(response["answer"]))
