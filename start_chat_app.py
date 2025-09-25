import pickle
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv


_template = """
You are given a conversation history and a follow-up question. 
Your task is to rephrase the follow-up question so that it becomes a clear, standalone question 
that can be understood without needing the prior context. 

Chat History:
{chat_history}

Follow-up Question:
{question}

Rephrased Standalone Question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


template = """
You are a helpful and friendly AI assistant with expertise in machine learning and technical topics. 
Your job is to provide clear, accurate, and conversational answers based strictly on the given context. 
- If the context does not contain the answer, politely say you don’t know instead of making things up.  
- Use a natural and approachable tone, like you are explaining to a colleague.  
- Format the response in Markdown for better readability.  

Question:
{question}

=========
Context:
{context}
=========

Answer in Markdown:
"""

QA = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0.3,model = 'gpt-4o-mini')
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3,"threshold":0.3}),
        condense_question_prompt = CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA},
        return_source_documents=True,
        verbose=True,
    )
    return qa_chain


if __name__ == "__main__":
    # with open("faiss_store.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_store", embeddings,allow_dangerous_deserialization=True)
    qa_chain = get_chain(vectorstore)
    print("Chat with the Strikingly.com bot:")
    print("Your question:")
    question = input()
    result = qa_chain({"question": question, "chat_history": []})
    print(f"AI: {result['answer']}")
    print("Sources:")
    for doc in result['source_documents']:
        print(doc.metadata['source'])