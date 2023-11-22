from langchain.llms import GooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

#LLM
llm = GooglePalm(google_api_key= os.environ["GOOGLE_API_KEY"], temperature =0.1)

#EMBEDDINGS AND VDB

# Create an instance of HuggingFaceInstructEmbeddings
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index" #Filepath for the vector DB
# Use the embeddings instance with FAISS.from_documents


#Store the action in a file
def create_vector_db():
    loader = PyPDFLoader("Quantum Computation and Quantum Information.pdf")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path) #Save it to a local file

# pdf_files = ["Quantum Computation and Quantum Information.pdf", "AnotherDocument.pdf", "MoreData.pdf"]
# create_vector_db(pdf_files)
# def create_vector_db(pdf_files): for multiple PDF
#     all_data = []
#
#     # Load and process each PDF file
#     for pdf_file in pdf_files:
#         loader = PyPDFLoader(pdf_file)
#         data = loader.load()
#         all_data.extend(data)  # Combine data from all PDFs
#
#     # Create a new FAISS index from the combined data
#     vectordb = FAISS.from_documents(documents=all_data, embedding=embeddings)
#     vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings) #load the filefromthe vector db and the embeddings
    retriever = vectordb.as_retriever()
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I am not sure about the answer please ask the teacher !" Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type (llm=llm,
                chain_type ='stuff', #You can choose MapReduce
                retriever =retriever,
                input_key="query",
                return_source_documents=True,
                chain_type_kwargs={"prompt":PROMPT})
    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("What is quantum computing?"))