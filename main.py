from dotenv import load_dotenv
from langchain import hub
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv('.env')
 
loaders = [
    TextLoader("docs/en/carl-oskar-bohlin.txt", encoding='utf8'),
    TextLoader("docs/en/tobias-billstrom.txt", encoding='utf8'),
    TextLoader("docs/en/ulf-kristersson.txt", encoding='utf8')
]

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 100)
docs = []
for loader in loaders:
    docs.extend(loader.load())

splits = text_splitter.split_documents(loader.load())

# Embed and store splitted documents
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
Chroma.from_documents(splits, embedding_function, persist_directory="./chroma_db")
print("Local Chroma DB ./chroma_db has been created")

# Open vectorstore for context providing to LLM
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
retriever = vectorstore.as_retriever()

def get_rag_chain(rag_prompt):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    )
    return rag_chain

def execute_query_and_print_result(rag_chain, query):
    result = rag_chain.invoke(query)
    print(result)

# Prompt
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = get_rag_chain(rag_prompt)

execute_query_and_print_result(rag_chain, "Who of the speakers is Prime Minister?")
execute_query_and_print_result(rag_chain, "When will Sweden reactivate civil service according to who?")
execute_query_and_print_result(rag_chain, "What actions is the Prime Minister talking about?")
execute_query_and_print_result(rag_chain, "What is the name of the Civil Defence Minister?")
execute_query_and_print_result(rag_chain, "What is the speech by the Prime Minister about?")
execute_query_and_print_result(rag_chain, "What is the speech by Carl-Oskar Bohlin about?")

# Prompt
template = """You are an helpful assistant who answers in Haiku. If you don't know the answer, make up an answer. Use three sentences maximum and keep the answer verbose and poetic.
Question: {question}
Context: {context}
Answer:
"""
haiku_rag_prompt = ChatPromptTemplate.from_template(template)
haiku_rag_chain = get_rag_chain(haiku_rag_prompt)
# Specifik rag_chain ska inte veta vad Sebastians favoritmat är och svara därefter.
execute_query_and_print_result(rag_chain, "What is Sebastians favourite food?")
execute_query_and_print_result(haiku_rag_chain, "What is Sebastians favourite food?")
