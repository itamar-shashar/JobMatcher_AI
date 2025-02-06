# Job Matcher AI - Your Smart AI-Powered Job Search Assistant

🔍 Finding the right job shouldn't feel like searching for a needle in a haystack. **Job Matcher AI** is an AI-powered job search assistant that leverages **Retrieval-Augmented Generation (RAG)**, **vector search**, **LLM-driven query refinement**, and **intelligent reranking** to deliver highly relevant job recommendations.

---


## 📖 Table of Contents

- [💡 Overview](#overview)
- [⚙️ How to Use the Job Matcher AI App](#how-to-use-the-job-matcher-ai-app)
- [🛠️ System Architecture](#system-architecture)
- [🌎 Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [📡 Web Scraping - Collecting Job Listings](#web-scraping---collecting-job-listings)
  - [📊 Data Preprocessing and Vector Database Setup](#data-preprocessing-and-vector-database-setup)
- [🔮 Future Improvements](#notes--future-improvements)
- [📜 License](#license)

---

## Overview

**Job Matcher AI** uses state-of-the-art AI models to help users find the most relevant job listings by refining user queries, searching vector databases, and ranking results intelligently. 

It features:

✅ **Real-time job retrieval** using a combination of LLMs and vector search.

✅ **AI-powered query refinement** for better search accuracy.

✅ **Semantic Graph Chunking** for improved text segmentation.

✅ **Reranking with cross-encoders** to ensure top-quality results.

✅ **Interactive UI** powered by **Streamlit**.

---

## How to Use the Job Matcher AI App

### 📦 Installation & Requirements
Ensure all dependencies are installed using the `requirements.txt` file.

Run this command on your IDE terminal:
```bash
pip install -r requirements.txt
```

### 🎯 Running the Job Matcher AI App
#### **📂 Files Needed:**
- `main_app_code.py` - The core backend logic that retrieves, ranks, and refines job matches.
- `streamlit_app.py` - The Streamlit UI to interact with the system.

#### **▶️ Setup Instructions:**
1. Ensure `main_app_code.py` and `streamlit_app.py` are in the same directory.
2. Obtain API keys for **Pinecone** and **Cohere**, and store them in a `.env` file (not included in this repository).
3. Open a terminal, navigate to the app directory, and run:
   ```bash
   streamlit run streamlit_app.py
   ```
4. The app will launch on a local host.
5. ⏳ **Expect an initial load time of 20-30 seconds.**

---

## System Architecture

🖼️ *(Insert a diagram here showcasing the entire RAG pipeline, including query refinement, retrieval, reranking, and response generation.)*

---

## 🌎 Data Collection and Preprocessing (For Developers and Enthusiasts)

### 📡 Web Scraping - Collecting Job Listings
#### **📝 Files:**
- `indeed_scraper.py` - The job listing scraper.
- `run_scraper.py` - Runs the scraper in a loop every few hours.

#### **🔑 Requirements:**
- Requires **Bright Data** credentials (**USERNAME** and **PASSWORD**) stored in the `.env` file.
- The scraper runs automatically when `run_scraper.py` is executed.

### 📊 Data Preprocessing and Vector Database Setup
#### **🔍 Files:**
- `analys_preprocess_and_vectordb.ipynb` - Jupyter Notebook for:
  - 🛠️ **Data cleaning** and **preprocessing** with **Apache Spark**.
  - 🔡 **Text normalization** and **feature engineering** (e.g., adding education level, filling missing values).
  - 🧩 **Chunking job descriptions** using **Semantic Graph Chunking**.
  - 📦 **Storing job listings in Pinecone Vector DB**.
- `semantic_graph_chunker.py` - Standalone script for **Semantic Graph Chunking**.

#### **⚙️ How to Run:**
- Best run on **Databricks** due to large-scale data processing.
- 📂 Upload the **parquet dataset** (`combined_data_80k.parquet` or `combined_data_190k.parquet`) to Databricks.
- Set the **data directory, Pinecone API key, and index name** in the first cell of the notebook.
- ⏳ **Running this step takes time, but it is not necessary to use the app**, as the data is already stored in Pinecone.

---

## 🔮 Notes & Future Improvements
- 🔑 The `.env` file containing API keys is **not included** in this repository and must be obtained separately.
- 📌 The preprocessing step is **not required** to run the app, as the job data is already indexed.
- ⚡ Future work will focus on **optimizing speed**, **improving prompt engineering**, and **scaling the system** to integrate with real-time job boards.

---

## 📜 License
This project is intended for educational and research purposes.

💬 For any inquiries, please open an issue or reach out to the contributors.

