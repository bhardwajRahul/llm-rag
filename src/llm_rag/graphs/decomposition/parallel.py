import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.markdown import Markdown
from rich.pretty import Pretty

from llm_rag import llm, retriever

decomposition_prompt_template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered sequentially.
Generate multiple search queries related to: {question}"""
decomposition_prompt = ChatPromptTemplate.from_template(decomposition_prompt_template)


sub_question_prompt_template = """Answer the following question based on this context:

{context}

Question: {question}
"""
sub_question_prompt = ChatPromptTemplate.from_template(sub_question_prompt_template)


rag_prompt_template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)


def format_qa_pair(question: str, answer: str) -> str:
    return f"Question: {question}  \nAnswer: {answer}\n\n\n"


class State(TypedDict):
    question: str
    generated_sub_questions: list[str]
    qa_pairs: Annotated[list[dict[str, str]], operator.add]
    context: list[Document]
    answer: str


class RetrieverState(TypedDict):
    generated_sub_question: str


def generate_sub_questions(query: str, config: RunnableConfig) -> list[str]:
    max_generated_sub_questions_count = config["configurable"].get(
        "max_generated_sub_questions_count", 3
    )

    class SubQuestionsGenerator(BaseModel):
        sub_questions: list[str] = Field(
            ...,
            description="List of generated sub-problems / sub-questions",
            max_items=max_generated_sub_questions_count,
        )

    structured_llm = llm.with_structured_output(
        SubQuestionsGenerator, method="function_calling"
    )
    chain = decomposition_prompt | structured_llm
    response = chain.invoke(query)
    questions = response.sub_questions

    return {"generated_sub_questions": questions}


def assign_sub_questions(state: State):
    return [
        Send("answer_sub_question", {"generated_sub_question": sub_question})
        for sub_question in state["generated_sub_questions"]
    ]


def answer_sub_question(state: RetrieverState):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | sub_question_prompt
        | llm
        | StrOutputParser()
    )

    question = state["generated_sub_question"]
    answer = chain.invoke(question)
    return {"qa_pairs": [{question: answer}]}


def aggregate_qa_pairs(state: State):
    context = ""

    for qa_pair in state["qa_pairs"]:
        [(question, answer)] = qa_pair.items()
        context += format_qa_pair(question, answer)

    return {"context": context}


def generate_answer(state: State):
    chain = rag_prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"question": state["question"], "context": state["context"]}
    )
    return {"answer": response}


class ConfigSchema(BaseModel):
    max_generated_sub_questions_count: int = Field(default=3, gt=1)


graph_builder = StateGraph(State, ConfigSchema)

graph_builder.add_node("generate_sub_questions", generate_sub_questions)
graph_builder.add_node("answer_sub_question", answer_sub_question)
graph_builder.add_node("aggregate_qa_pairs", aggregate_qa_pairs)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "generate_sub_questions")
graph_builder.add_conditional_edges(
    "generate_sub_questions", assign_sub_questions, ["answer_sub_question"]
)
graph_builder.add_edge("answer_sub_question", "aggregate_qa_pairs")
graph_builder.add_edge("aggregate_qa_pairs", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    query = "What are the main components of an LLM-powered autonomous agent system?"
    config = {
        "configurable": {
            "max_generated_sub_questions_count": 5,
        }
    }
    response = graph.invoke(
        {"question": query},
        config=config,
    )

    rprint(Pretty(response, max_depth=2))
    rprint(Markdown(response["answer"]))
