from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mcp.server.fastmcp import FastMCP


# Define MCP server ===
mcp = FastMCP(name="Tax Law Agent")

# Load and index the PDF ===
reader = PdfReader("TaxProceduresAct29of2015.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))



@mcp.tool()
def ask_tax_question(question: str) -> str:
    """Search Tax Procedures Act PDF for a given question."""
    query_vec = model.encode([question]).astype("float32")
    D,I = index.search(query_vec,k=3)
    context = "\n\n".join([chunks[i] for i in I[0]])
    return context



if __name__ == "__main__":
    mcp.run(transport = "stdio")
