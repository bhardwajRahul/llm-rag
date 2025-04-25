from typing import Literal, TypedDict

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

retrieval_prompt_template = """You are grader assistant assessing the need to retrieve additional documents to answer the user's question.
If you are sure that all the necessary data is available, then you do not need to retrieve additional documents.
Give a binary score to indicate whether retrieval is required.

User question:
{question}
"""

rag_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

answer_prompt_template = """Answer the following question:

Question: {question}
"""

no_answer_prompt = "I don't have an answer to the question."

relevance_grading_prompt_template = """You are a grader assessing relevance of a retrieved document to a user question.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score to indicate whether the document is relevant to the question.

Retrieved document:
{document}

User question:
{question}
"""

hallucinations_grading_prompt_template = """You are a grader assessing whether an LLM answer is grounded in / supported by a set of retrieved facts.
Give a binary score whether the answer is grounded in / supported by the set of facts.

Set of facts:
{context}

LLM answer:
{answer}
"""

answer_grading_prompt_template = """You are a grader assessing whether an answer addresses / resolves a question. 
Give a binary score whether the answer resolves the question.

User question:
{question}

LLM answer:
{answer}
"""

query_rewriting_prompt_template = """You a question re-writer that converts an input question to a better version that is optimized for web search. 
Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:
{question}

Formulate an improved question."""


def format_docs(docs: list[Document]) -> list[str]:
    return "\n\n".join(doc.page_content for doc in docs)


class RetrievalGrade(BaseModel):
    """Check if retrieval of additional documents is required."""

    chain_of_thought: str = Field(
        ...,
        description="Step by step reasoning to check if retrieval of additional documents is required",
    )
    is_required: bool = Field(
        description="Retrieval of additional documents is required"
    )


retrieval_grader_llm = llm.with_structured_output(
    RetrievalGrade, method="function_calling"
)


class RelevanceGrade(BaseModel):
    """Relevance check on retrieved document."""

    chain_of_thought: str = Field(
        ...,
        description="Step by step reasoning to check if the document is relevant to the question",
    )
    is_relevant: bool = Field(description="Document is relevant to the question")


relevance_grader_llm = llm.with_structured_output(
    RelevanceGrade, method="function_calling"
)


@chain
def grade_document_relevance(document, question):
    relevance_grading_prompt = relevance_grading_prompt_template.format(
        document=document, question=question
    )
    response = relevance_grader_llm.invoke(
        [HumanMessage(content=relevance_grading_prompt)]
    )
    return response


class HallucationsGrade(BaseModel):
    """Hallucination check in generated answer."""

    chain_of_thought: str = Field(
        ...,
        description="Step by step reasoning to check if the answer is grounded in the facts",
    )
    is_grounded: bool = Field(description="Answer is grounded in the facts")


hallucations_grader_llm = llm.with_structured_output(
    HallucationsGrade, method="function_calling"
)


class AnswerGrade(BaseModel):
    """Check if answer addresses the question."""

    chain_of_thought: str = Field(
        ...,
        description="Step by step reasoning to check if the answer addresses the questions",
    )
    is_useful: bool = Field(description="Answer addresses the question")


answer_grader_llm = llm.with_structured_output(AnswerGrade, method="function_calling")


class SearchQuery(BaseModel):
    """Question optimization for search."""

    chain_of_thought: str = Field(
        ..., description="Step by step reasoning to optimize query for search"
    )
    search_query: str = Field(description="Optimized search query")


search_llm = llm.with_structured_output(SearchQuery, method="function_calling")


class State(TypedDict):
    question: str
    retrieval_grade: RetrievalGrade
    documents: list[Document]
    relevance_grades: list[RelevanceGrade]
    generation: str
    hallucinations_grade: HallucationsGrade
    context: list[Document]
    answer_grade: AnswerGrade
    answer: str


def grade_retrieval(state: State):
    question = state["question"]
    retrieval_prompt = retrieval_prompt_template.format(question=question)
    retrieval_grade = retrieval_grader_llm.invoke(retrieval_prompt)
    return {"retrieval_grade": retrieval_grade}


def decide_to_retrieve(state: State) -> Literal["retrieve", "generate_answer"]:
    retrieval_grade = state["retrieval_grade"]

    if retrieval_grade.is_required:
        return "retrieve"
    else:
        return "generate_answer"


def retrieve(state: State):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}


def grade_documents(state: State):
    question = state["question"]
    documents = state["documents"]

    relevance_grades = grade_document_relevance.batch(documents, question=question)
    filtered_documents = [
        document
        for (document, relevance_grade) in zip(documents, relevance_grades)
        if relevance_grade.is_relevant
    ]

    return {"context": filtered_documents, "relevance_grades": relevance_grades}


def check_documents_relevance(
    state: State,
) -> Literal["generate_rag_answer", "generate_no_answer"]:
    filtered_documents = state["context"]

    if len(filtered_documents) > 0:
        return "generate_rag_answer"
    else:
        return "generate_no_answer"


def generate_rag_answer(state: State):
    docs_content = format_docs(state["context"])
    rag_prompt = rag_prompt_template.format(
        question=state["question"], context=docs_content
    )
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"answer": response.content}


def generate_answer(state: State):
    answer_prompt = answer_prompt_template.format(question=state["question"])
    response = llm.invoke([HumanMessage(content=answer_prompt)])
    return {"answer": response.content}


def generate_no_answer(state: State):
    return {"answer": no_answer_prompt}


def grade_hallucinations(state: State):
    filtered_documents = state["context"]
    answer = state["answer"]
    hallucinations_grading_prompt = hallucinations_grading_prompt_template.format(
        context=filtered_documents, answer=answer
    )
    hallucinations_grade = hallucations_grader_llm.invoke(hallucinations_grading_prompt)
    return {"hallucinations_grade": hallucinations_grade}


def check_hallucinations(
    state: State,
) -> Literal["grade_answer", "generate_rag_answer"]:
    hallucinations_grade = state["hallucinations_grade"]

    if hallucinations_grade.is_grounded:
        return "grade_answer"
    else:
        return "generate_rag_answer"


def grade_answer(state: State):
    question = state["question"]
    answer = state["answer"]
    answer_grading_prompt = answer_grading_prompt_template.format(
        question=question, answer=answer
    )
    answer_grade = answer_grader_llm.invoke(answer_grading_prompt)
    return {"answer_grade": answer_grade}


def check_answer(state: State) -> Literal["__end__", "rewrite_query"]:
    answer_grade = state["answer_grade"]

    if answer_grade.is_useful:
        return "__end__"
    else:
        return "rewrite_query"


def rewrite_query(state: State):
    question = state["question"]
    query_rewriting_prompt = query_rewriting_prompt_template.format(question=question)
    response = search_llm.invoke(query_rewriting_prompt)
    return {"question": response.search_query}


graph_builder = StateGraph(State)

graph_builder.add_edge(START, "grade_retrieval")
graph_builder.add_node("grade_retrieval", grade_retrieval)
graph_builder.add_conditional_edges("grade_retrieval", decide_to_retrieve)

graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge("generate_answer", END)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_conditional_edges("grade_documents", check_documents_relevance)

graph_builder.add_node("generate_rag_answer", generate_rag_answer)
graph_builder.add_edge("generate_rag_answer", "grade_hallucinations")
graph_builder.add_node("grade_hallucinations", grade_hallucinations)
graph_builder.add_conditional_edges("grade_hallucinations", check_hallucinations)

graph_builder.add_node("generate_no_answer", generate_no_answer)
graph_builder.add_edge("generate_no_answer", END)

graph_builder.add_node("grade_answer", grade_answer)
graph_builder.add_conditional_edges("grade_answer", check_answer)

graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_edge("rewrite_query", "retrieve")

graph = graph_builder.compile()


if __name__ == "__main__":
    queries = [
        "What are common types of agent memory?",
        "What are recent types of adversarial attacks in LLM?",
        "How does the AlphaCodium paper work?",
    ]

    for query in queries:
        response = graph.invoke({"question": query})
        rprint(Pretty(response))
        rprint(Markdown(response["answer"]))
