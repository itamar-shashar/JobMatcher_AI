# Job Matcher AI - Your Smart AI-Powered Job Search Assistant

## 📖 Table of Contents
- [Overview](#overview)
- [How to Use the Job Matcher AI App](#how-to-use-the-job-matcher-ai-app)
- [System Architecture](#system-architecture)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing-for-developers-and-enthusiasts)
  - [Web Scraping - Collecting Job Listings](#web-scraping---collecting-job-listings)
  - [Data Preprocessing and Vector Database Setup](#data-preprocessing-and-vector-database-setup)
- [Installation & Requirements](#installation--requirements)
- [Notes & Future Improvements](#notes--future-improvements)
- [License](#license)


## 🌟 Overview
Finding the right job shouldn't feel like searching for a needle in a haystack. **Job Matcher AI** is an AI-powered job search assistant that leverages **Retrieval-Augmented Generation (RAG)**, **vector search**, **LLM-driven query refinement**, and **intelligent reranking** to deliver highly relevant job recommendations.

## 🚀 How to Use the Job Matcher AI App
### Running the Job Matcher AI App
#### **Files Needed:**
- `main_app_code.py` - The core backend logic that retrieves, ranks, and refines job matches.
- `streamlit_app.py` - The Streamlit UI to interact with the system.

#### **Setup Instructions:**
1. Ensure `main_app_code.py` and `streamlit_app.py` are in the same directory.
2. Obtain API keys for **Pinecone** and **Cohere**, and store them in a `.env` file (not included in this repository).
3. Open a terminal, navigate to the app directory, and run:
   ```bash
   streamlit run streamlit_app.py
   ```
4. The app will launch on a local host.
5. Expect an initial load time of **20-30 seconds**.

## 🏗️ System Architecture
![System Architecture](path/to/system_architecture_image.png)

## 🔍 Data Collection and Preprocessing (For Developers and Enthusiasts) (For Developers and Enthusiasts)
If you're interested in how the job listings are collected, processed, and indexed, this section provides a deep dive into our data pipeline.

### Web Scraping - Collecting Job Listings
#### **Files:**
- `indeed_scraper.py` - The job listing scraper.
- `run_scraper.py` - Runs the scraper in a loop every few hours.

#### **Requirements:**
- Requires **Bright Data** credentials (**USERNAME** and **PASSWORD**) stored in the `.env` file.
- The scraper runs automatically when `run_scraper.py` is executed.

### Data Preprocessing and Vector Database Setup
#### **Files:**
- `analys_preprocess_and_vectordb.ipynb` - Jupyter Notebook for:
  - Data cleaning and preprocessing with **Apache Spark**.
  - Text normalization and feature engineering (e.g., adding education level, filling missing values).
  - Chunking job descriptions using **Semantic Graph Chunking**.
  - Storing job listings in **Pinecone Vector DB**.
- `semantic_graph_chunker.py` - Standalone script for **Semantic Graph Chunking**.

#### **How to Run:**
- Best run on **Databricks** due to large-scale data processing.
- Upload the **parquet dataset** (`combined_data_80k.parquet` or `combined_data_190k.parquet`) to Databricks.
- Set the **data directory, Pinecone API key, and index name** in the first cell of the notebook.
- Running this step takes time, but it is **not necessary** to use the app, as the data is already stored in Pinecone.

## 🛠️ Installation & Requirements
Ensure all dependencies are installed using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## 🚀 Notes & Future Improvements
- The `.env` file containing API keys is **not included** in this repository and must be obtained separately.
- The preprocessing step is **not required** to run the app, as the job data is already indexed.
- Future work will focus on **optimizing speed**, **improving prompt engineering**, and **scaling the system** to integrate with real-time job boards.

## 📜 License
This project is intended for educational and research purposes.

For any inquiries, please open an issue or reach out to the contributors.

