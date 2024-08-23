import pandas as pd
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

def main():
    # Initialize the LLM with the base URL
    llm = Ollama(
        model="jackalope:latest",
        temperature=0,
        base_url='http://host.docker.internal:11434'
    )
    
    print("libmagic and python-magic are installed and working correctly!")

    # Load the CSV file using pandas
    df = pd.read_csv("Salary_Data_Based_country_and_race.csv")

    # Convert the DataFrame to a list of strings
    documents = df.apply(lambda row: row.to_string(), axis=1).tolist()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(' '.join(documents))
    
    # Ensure the model is available
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Create embeddings using the correct model name
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Create the FAISS vector store
    salary_base = FAISS.from_texts(text_chunks, embeddings)
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=salary_base.as_retriever()
    )
    
    # Ask some questions
    questions = [
        "What is this document about?",
        "What are the countries present?",
        "What are the Titles present in the document?"
    ]
    
    for question in questions:
        response = qa_chain.invoke({"query": question})
        print(f"Question: {question}")
        print(f"Answer: {response['result']}\n")

if __name__ == "__main__":
    main()