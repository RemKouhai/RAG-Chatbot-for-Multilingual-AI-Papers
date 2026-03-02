# 🧠 RAG Chatbot for Multilingual AI Papers

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)

An Advanced Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex academic papers (PDFs). This project is heavily optimized to run **100% locally** on severely constrained hardware (e.g., a single NVIDIA T4 16GB GPU) without sacrificing the reasoning capabilities of a 7-billion parameter LLM.

## ✨ Key Features & Technical Highlights

* **SOTA Academic Parsing:** Utilizes **IBM Docling** to perfectly extract texts from complex multi-column academic PDFs while preserving crucial layout elements like mathematical formulas and data tables.
* **Two-Stage Retrieval System:** * *Stage 1 (Hybrid Retrieval):* Uses `BAAI/bge-m3` to fetch the Top 10 context chunks.
    * *Stage 2 (Cross-Encoder Reranking):* Uses `BAAI/bge-reranker-base` to refine the results down to the Top 3 most relevant chunks, filtering out semantic noise.
* **Ultra-Low Latency Tensor Storage:** Completely bypasses traditional Vector Databases (like Chroma/Pinecone). Employs **Low-level In-Memory Tensor Storage** (`.pt` and `.json`) loaded directly into VRAM via PyTorch, achieving <30ms retrieval latency.
* **Hardware-Efficient LLM (4-bit NF4):** Deploys **Qwen 2.5 7B Instruct** using `bitsandbytes` 4-bit NormalFloat (NF4) quantization. Fits a massive 7B model into just ~5.5GB of VRAM.
* **Interactive Web UI:** Built with **Streamlit** and tunneled via **ngrok**, providing a real-time conversational interface with transparent source citations.

## 🏗️ System Architecture

The inference pipeline is designed as a closed-loop, two-stage RAG architecture:

```mermaid
graph LR
    classDef layer1 fill:#eef2ff,stroke:#818cf8,stroke-width:1.5px,color:#1e1b4b;
    classDef layer2 fill:#f5f3ff,stroke:#a78bfa,stroke-width:1.5px,color:#2e1065;
    classDef db fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px,color:#0f172a;
    classDef model fill:#ecfdf5,stroke:#34d399,stroke-width:1.5px,color:#064e3b;

    subgraph L1["🌐 Layer 1: Access & Deployment"]
        direction LR
        U(["fa:fa-user User"]):::layer1
        Ngrok(["fa:fa-shield-halved Ngrok Tunnel"]):::layer1
        UI(["fa:fa-desktop Streamlit UI"]):::layer1
    end

    subgraph L2["🧠 Layer 2: Core RAG Inference Pipeline"]
        direction LR

        subgraph Retrieval["⚙️ Two‑Stage Retrieval"]
            direction TB
            Retriever(["fa:fa-microchip BAAI/bge-m3"]):::model
            DB(["fa:fa-database Local Tensor DB"]):::db
            Retriever -.->|Query| DB
        end

        Reranker(["fa:fa-scale-balanced Reranker"]):::model
        Prompt(["fa:fa-pen-nib Prompt Builder"]):::model
        Generator(["fa:fa-robot Qwen 2.5 7B (NF4)"]):::model
    end

    U -->|Query| Ngrok
    Ngrok -->|Access| UI
    UI -->|Query| Retriever
    UI -.->|Query| Reranker
    UI -.->|Query| Prompt

    Retriever ==|Top 10| Reranker
    Reranker ==|Top 3| Prompt

    Prompt -->|Context| Generator
    Generator ==|Response| UI

    style L1 fill:#f5f7ff,stroke:#c7d2fe,stroke-dasharray:4 4
    style L2 fill:#faf5ff,stroke:#ddd6fe,stroke-dasharray:4 4
    linkStyle 2,3,4 stroke:#2563eb,stroke-width:2.5px
    linkStyle 5,6 stroke:#ea580c,stroke-width:3px
    linkStyle 8 stroke:#ca8a04,stroke-width:3px