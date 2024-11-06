import langchain
import langchain_experimental
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def cvs_loader (path):
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(
        file_path=path,
        csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['Index', 'Height', 'Weight']
    })
    docs = loader.load()
    return docs

def pdf_loader (path):

    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(
        path,
    )
    docs = loader.load()
    return docs

def txt_loader (path):
    with open(path) as f:
        text = f.read()
        return text
    

text = txt_loader('LeyesColTransport2.txt')
text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="gradient"
)
docs = text_splitter.create_documents([text])
print(docs[0].page_content)
print(len(docs))