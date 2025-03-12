from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from llm_rag.graphs.utils import load_documents, prepare_vectorstore

load_dotenv(find_dotenv())


llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
docs = load_documents()
vectorstore = prepare_vectorstore(docs, embeddings)
retriever = vectorstore.as_retriever()
