from dotenv import load_dotenv
import os

### Build Index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


from langchain_openai import ChatOpenAI


### Search
from langchain_tavily import TavilySearch

#

# load .env file

env_path = "C:/Users/fyfig/Desktop/code/adaptive-ARG/.env"  # 你的 .env 路径
is_loaded = load_dotenv(dotenv_path=env_path)
# load api_key config
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")

print(openai_key)
print(tavily_key)
print(USER_AGENT)
### 1. Create Index

#Set embeddings
embd = OpenAIEmbeddings(
    base_url="https://aihubmix.com/v1",
)

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()