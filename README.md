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

## 📊 Evaluation & Benchmarks
The system underwent rigorous component-wise evaluation (LLM-as-a-Judge) using the **RAGAS** framework on a test set of 200+ academic queries:
* **Retrieval Recall@1:** `92.4%` (BGE-M3 Hybrid Search)
* **Reranker MRR (Mean Reciprocal Rank):** `0.967` (BGE-Reranker-base)
* **End-to-end Inference Speed:** ~11 tokens/second.
