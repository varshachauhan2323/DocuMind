📚 DocuMind

RAG-Powered PDF Intelligence Assistant

Upload your documents. Ask questions naturally. Get grounded answers with source citations.

<p align="center">
  <a href="https://documind-rag-pdf-assistant.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-DocuMind-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/varshachauhan2323/DocuMind">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=flat-square">
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=flat-square&logo=supabase&logoColor=white">
  <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square">
</p>

🌐 Live Application

👉 Open DocuMind

🧩 Problem

Working with large PDFs manually is slow and frustrating.

Finding one specific answer can require:

reading hundreds of pages

searching through multiple documents

remembering where information appeared

comparing sections manually

verifying whether an AI-generated answer is actually supported

Traditional PDF chatbots often solve only the last part — chatting.

DocuMind focuses on the complete workflow:

PDFs
  ↓
Document Processing
  ↓
Hybrid Retrieval
  ↓
Reranking
  ↓
Grounded Generation
  ↓
Answer + Sources + Verification

💡 What is DocuMind?

DocuMind is an intelligent PDF question-answering and document analysis application built using Retrieval-Augmented Generation (RAG).

Users can upload one or multiple PDFs and interact with them through natural language.

Instead of sending the complete document to an LLM, DocuMind:

extracts the document text

splits it into meaningful chunks

creates vector embeddings

indexes the content

retrieves relevant evidence

reranks the retrieved chunks

builds a focused context

generates a grounded response

provides source/page information for verification

This makes the system more focused, efficient, and transparent.

✨ Key Features

Feature

Description

📄 Multi-PDF Upload

Upload and work with multiple PDF documents

💬 Conversational Q&A

Ask questions naturally and continue with follow-ups

🔎 Hybrid Retrieval

Combines semantic and keyword retrieval

🧠 CrossEncoder Reranking

Reranks retrieved chunks for better relevance

🔄 Query Rewriting

Converts conversational questions into retrieval-friendly queries

🛡️ Grounded Responses

Uses retrieved document evidence instead of relying only on model memory

📑 Page Citations

Provides source/page references for verification

🔬 Citation Verification

Indicates Supported / Partially Supported / Unsupported evidence

⚖️ Compare Mode

Compare two documents using the RAG pipeline

👤 User Isolation

Separates user documents and conversation state

🔐 Authentication

Google OAuth + Email/Password

🔑 Password Recovery

Forgot-password and reset-password flow

📊 Analytics

Usage and retrieval information

⚡ Smart Re-indexing

Detects changed PDFs using content hashes

📤 Export

Export conversations/results as TXT or Markdown

📱 Responsive UI

Optimized for desktop and mobile

🏗️ System Architecture

                         ┌──────────────────────┐
                         │      User            │
                         │  Web / Mobile UI     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │    Application       │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │   Authentication │            │  PDF Processing  │
          │ Google / Supabase│            │   PyPDFLoader    │
          └──────────────────┘            └────────┬─────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │  Text Splitting  │
                                         │ 900 / 150 chunks │
                                         └────────┬─────────┘
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │    Embeddings    │
                                         │    MiniLM        │
                                         └────────┬─────────┘
                                                  │
                              ┌───────────────────┴───────────────────┐
                              │                                       │
                              ▼                                       ▼
                       ┌──────────────┐                        ┌──────────────┐
                       │    FAISS     │                        │     BM25     │
                       │   Semantic   │                        │   Keyword    │
                       │   Retrieval  │                        │   Retrieval  │
                       └──────┬───────┘                        └──────┬───────┘
                              │                                       │
                              └────────────────┬──────────────────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │ Hybrid Retrieval    │
                                    └──────────┬──────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │ CrossEncoder        │
                                    │ Reranking           │
                                    └──────────┬──────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │ Context Cleaning /  │
                                    │ Compression         │
                                    └──────────┬──────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │ Groq LLM            │
                                    │ openai/gpt-oss-20b  │
                                    └──────────┬──────────┘
                                               ▼
                              ┌────────────────────────────────┐
                              │ Answer + Citations + Quality  │
                              │ / Verification Information    │
                              └────────────────────────────────┘

🔬 RAG Pipeline

1. Document Ingestion

When a PDF is uploaded:

PDF
 ↓
PyPDFLoader
 ↓
Extracted Text
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Index + BM25 Index

Current chunking configuration:

Chunk Size      : 900 characters
Chunk Overlap   : 150 characters

The overlap helps maintain context between adjacent chunks.

2. Embedding Generation

DocuMind uses:

sentence-transformers/all-MiniLM-L6-v2

Each document chunk is converted into a numerical vector representation.

These vectors allow the application to retrieve text based on meaning, not only exact keyword matches.

3. Hybrid Retrieval

DocuMind combines:

FAISS

Used for semantic/vector similarity search.

BM25

Used for lexical/keyword matching.

                 User Query
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       FAISS                  BM25
   Semantic Search        Keyword Search
          │                     │
          └──────────┬──────────┘
                     ▼
              Candidate Chunks

Hybrid retrieval is particularly useful when a question contains both conceptual meaning and exact terms.

4. CrossEncoder Reranking

The initial retrieval stage may return several relevant candidates.

A CrossEncoder evaluates the relationship between:

Query ↔ Retrieved Chunk

and reranks the candidates so the most useful evidence appears first.

5. Context Construction

Before sending information to the LLM, DocuMind cleans and compresses the retrieved context.

The goal is:

More relevant evidence
        +
Less irrelevant text
        =
Better grounded response

6. Grounded Generation

The final context is passed to the Groq LLM:

openai/gpt-oss-20b

The model generates the answer using the retrieved document evidence.

🛡️ Grounded Answers

A major goal of DocuMind is to avoid confidently answering questions that are not supported by the uploaded documents.

The application therefore evaluates retrieval/citation quality and can classify evidence as:

✅ Supported
⚠️ Partially Supported
❌ Unsupported

This gives users an additional signal about how strongly the answer is backed by the source material.

📑 Source Citations

Where available, responses include source/page information.

Example:

The proposed architecture uses a hybrid retrieval strategy.

Source:
Document A — Page 7

This makes it easier for users to go back to the original PDF and verify the answer.

⚖️ Document Comparison

DocuMind includes a dedicated Compare workflow.

Users can select two documents and ask questions such as:

Compare the methodologies used in both documents.

What are the major differences between these reports?

Which document proposes a more detailed architecture?

The system retrieves evidence from the appropriate documents before generating the comparison.

💬 Conversational Memory

DocuMind supports follow-up questions.

For example:

User:
What is the proposed algorithm?

DocuMind:
The proposed algorithm is ...

User:
What are its limitations?

DocuMind:
The limitations of that algorithm are ...

The previous conversation helps resolve references such as:

it
they
this method
the above approach
its limitations

before retrieval.

🔐 Authentication & User Management

DocuMind supports:

Google OAuth

Users can authenticate using Google.

Email / Password

Users can create and use an email/password account.

Password Recovery

Users can request a reset link and set a new password through the recovery flow.

User Isolation

User-specific state is associated with the authenticated user, helping isolate:

documents

conversations

retrieval state

usage information

📊 Usage & Analytics

DocuMind includes application-level usage tracking.

Current application limit:

20 questions per user per day

The application also exposes retrieval-related information useful for understanding and debugging the RAG pipeline.

⚡ Smart Re-indexing

DocuMind uses a content hash for uploaded files.

Instead of assuming that a filename represents the same document, the application checks the actual file contents.

Upload PDF
    │
    ▼
Calculate Content Hash
    │
    ▼
Compare With Existing Hash
    │
    ├── Same ──► Reuse Index
    │
    └── Changed ► Re-index PDF

This reduces unnecessary embedding and indexing work.

📤 Export

Users can export generated content/conversations in:

.txt

.md

This makes it easier to save notes, research summaries, or document analysis results.

🧰 Technology Stack

Category

Technology

Language

Python

UI

Streamlit

LLM

Groq — openai/gpt-oss-20b

LLM Framework

LangChain

PDF Loader

PyPDFLoader

Embeddings

Sentence Transformers

Embedding Model

all-MiniLM-L6-v2

Vector Search

FAISS

Keyword Search

BM25

Reranking

CrossEncoder

Authentication

Supabase + Google OAuth

Database

Supabase PostgreSQL

Deployment

Streamlit Community Cloud

📁 Project Structure

DocuMind/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml
│
└── ...

app.py

Main application containing:

Streamlit UI

authentication

PDF processing

chunking

embeddings

FAISS retrieval

BM25 retrieval

reranking

conversational RAG

comparison mode

citations

analytics

export functionality

requirements.txt

Python dependencies required to run the project.

.gitignore

Protects local-only files such as:

.streamlit/secrets.toml
.env
venv/
__pycache__/
*.pyc

🚀 Getting Started

Prerequisites

Recommended:

Python 3.11+
Git

1️⃣ Clone the Repository

git clone https://github.com/varshachauhan2323/DocuMind.git
cd DocuMind

2️⃣ Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS / Linux

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure Secrets

Create:

.streamlit/secrets.toml

Example structure:

GROQ_API_KEY = "your-groq-api-key"

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-supabase-publishable-key"
SUPABASE_SERVICE_ROLE_KEY = "your-supabase-secret-key"

APP_URL = "http://localhost:8501"

[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "your-cookie-secret"
client_id = "your-google-client-id"
client_secret = "your-google-client-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

Never commit secrets.toml to GitHub.

5️⃣ Run the Application

streamlit run app.py

Open:

http://localhost:8501

☁️ Deployment

DocuMind is deployed using Streamlit Community Cloud.

Deployment flow:

GitHub
  ↓
Streamlit Community Cloud
  ↓
Configure Secrets
  ↓
Deploy
  ↓
Live Application

Production configuration

For production, configure:

Groq API key

Supabase URL

Supabase publishable/anonymous key

Supabase server-side secret

Google OAuth client ID

Google OAuth client secret

OAuth redirect URI

application URL

🔒 Security

DocuMind uses several security-sensitive credentials.

Never expose or commit:

GROQ_API_KEY
SUPABASE_SERVICE_ROLE_KEY
Google OAuth client secret
.streamlit/secrets.toml
.env

The Supabase service/secret key has elevated privileges and must remain server-side only.

Authentication recovery tokens should also never be logged or displayed.

🧪 Testing & Validation

Before committing changes:

python -m py_compile app.py

Recommended manual test checklist:

[ ] Email signup
[ ] Email login
[ ] Google login
[ ] Logout
[ ] Forgot password
[ ] Password reset
[ ] PDF upload
[ ] Multiple PDF upload
[ ] Normal Q&A
[ ] Follow-up questions
[ ] Compare mode
[ ] Page citations
[ ] Unsupported questions
[ ] Re-indexing after PDF changes
[ ] Export
[ ] Mobile layout

🎯 Example Questions

Try asking DocuMind:

What is the main objective of this document?

Summarize the methodology.

What are the key findings?

What limitations are mentioned?

Which algorithm is used?

Explain the proposed architecture.

What evidence supports this conclusion?

Which page discusses the results?

Compare the methodology of Document A and Document B.

What are the major differences between these documents?

🎓 Use Cases

📖 Education

Students can upload:

lecture notes

textbooks

assignments

research papers

and ask questions directly against the material.

🔬 Research

Researchers can:

analyze papers

find methodology details

locate results

compare multiple studies

💻 Technical Documentation

Developers can upload:

API documentation

specifications

technical reports

architecture documents

and ask targeted questions.

🏢 Business Documents

Teams can analyze:

reports

policies

proposals

internal documentation

while retaining source references.

⚠️ Limitations

DocuMind is designed for document-grounded Q&A, but some limitations remain:

Scanned/image-only PDFs may require OCR.

Complex PDF layouts can affect text extraction.

Tables and unusual formatting may not always extract perfectly.

Retrieval quality affects final answer quality.

LLM-generated answers should be verified for high-stakes decisions.

Authentication/email providers can enforce rate limits.

🔮 Future Improvements

Possible future enhancements:

OCR for scanned PDFs

Better table extraction

Image and diagram understanding

Persistent document storage

Background indexing

Streaming responses

Advanced citation highlighting

Retrieval evaluation dashboard

User-configurable retrieval parameters

Team workspaces

Conversation sharing

Larger-scale document collections

🌟 Why This Project?

DocuMind demonstrates the complete lifecycle of an AI-powered application:

Problem
  ↓
Document Ingestion
  ↓
Embeddings
  ↓
Vector Search
  ↓
Keyword Search
  ↓
Hybrid Retrieval
  ↓
Reranking
  ↓
Context Engineering
  ↓
LLM Generation
  ↓
Grounding & Citations
  ↓
Authentication
  ↓
User Isolation
  ↓
Deployment

It combines RAG, information retrieval, LLMs, authentication, databases, UI development, security, and cloud deployment into a single working application.

👩‍💻 Author

Varsha Chauhan

B.Tech — Computer Science & Engineering

🔗 Links

🚀 Live Demo: https://documind-rag-pdf-assistant.streamlit.app/

💻 GitHub: https://github.com/varshachauhan2323/DocuMind

<p align="center">
  <b>📚 DocuMind — Your documents. Your answers.</b>
  <br>
  Built with Python, Streamlit, LangChain, FAISS, Supabase & Groq.
</p>

