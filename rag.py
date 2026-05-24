from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
# 1. Load the existing vector store from disk

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 2. Create the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Create the llm
llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite-preview")

# 4. Prompt template

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context: {context}

Question: {question}

""")

# 5. LCEL chain
chain = (
    {"context" : retriever, "question" : RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
)

# 6. Ask something
while True:
        
    question = input("Question: ")
    if question.lower() == "none":
        break
    answer = chain.invoke(question)
    print(answer)