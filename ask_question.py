import argparse
import faiss
import os
import pickle
from langchain_openai import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()
key = os.environ.get("OPENAI_API_KEY")

parser = argparse.ArgumentParser(description='Strikingly.com Q&A')
parser.add_argument('question', type=str, help='Your question for Strikingly.com')
args = parser.parse_args()

embeddings = OpenAIEmbeddings(openai_api_key=key,)
store = FAISS.load_local("faiss_store", embeddings,allow_dangerous_deserialization=True)

chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0, verbose=True), vectorstore=store, verbose=True)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")