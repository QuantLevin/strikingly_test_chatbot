import argparse
import faiss
import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain,ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser(description='Strikingly.com Q&A')
parser.add_argument('question', type=str, help='Your question for Strikingly.com')
args = parser.parse_args()

embeddings = OpenAIEmbeddings()
store = FAISS.load_local("faiss_store", embeddings,allow_dangerous_deserialization=True)

# New version vector db qa chain for Task 2
chain_conversational = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, verbose=True),
    retriever=store.as_retriever(),
    return_source_documents=True,
    verbose=True
)
chat_history = []
result = chain_conversational.invoke({"question": args.question, "chat_history": chat_history})


print(f"Answer: {result['answer']}")
print("Sources:")
for doc in result['source_documents']:
    print(doc.metadata['source'])