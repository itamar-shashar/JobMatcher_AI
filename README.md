<h1 align="center" style="font-size: 10em;">Job Matcher AI - Your Smart AI-Powered Job Search Assistant</h1>

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

### 🛠️ Preparations
Store the files in the `app` folder, the `requierments.txt` and the .env file (need to get access) in the same directory

### 📦 Installation & Requirements
Make sure Python 3.11 or higher is installed on your computer.
Ensure all dependencies are installed using the `requirements.txt` file.

Run these commands in your IDE terminal (it is recommended to create and activate a new virtual environment first):
```bash
pip install -r requirements.txt
```
And after all libraries are installed, run this comand to install the SpaCy model:

On Windows:
```bash
python -m spacy download en_core_web_sm
```
On Mac:
```bash
python3 -m spacy download en_core_web_sm
```

### 🎯 Running the Job Matcher AI App
#### **📂  Files You Need to Store in the Same Directory:
1. The files from the from the [app](https://github.com/itamar-shashar/JobMatcher_AI/tree/main/app) folder:**
- `main_app_code.py` - The core backend logic that retrieves, ranks, and refines job matches.
- `streamlit_app.py` - The Streamlit UI to interact with the system.
2. The `requirements.txt` file.
3. The `.env` file (need to get access).

#### **▶️ Setup Instructions:**
1. Make sure you get access to the `.env` file that contains the API keys for **Pinecone** and **Cohere**, as it is required to run the application. Store it in the same directory as the app
2. Open a terminal on your IDE, navigate to the app directory, and run:
   ```bash
   streamlit run streamlit_app.py --server.fileWatcherType none
   ```
3. The app will launch on a local host.
4. ⏳ **Expect an initial load time of 20-30 seconds.**

---
<h1 align="center" style="font-size: 3em; margin-bottom: 20px;">
    🔍 Extra Information About Job Matcher AI For Developers and Enthusiasts
</h1>

<p align="center" style="font-size: 1.1em; color: gray;">
    The following sections are not required to run the app. This is simply to enrich your knowledge about the entire process.
</p>


## System Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/0f0750ba-da32-424f-9ee2-e3487c4e6ee0" alt="System Architecture">
</p>

---

## Data Collection and Preprocessing 
### Web Scraping - Collecting Job Listings
#### **📝 Files (from the [web scraping](https://github.com/itamar-shashar/JobMatcher_AI/tree/main/web%20scraping) folder):**
- `indeed_scraper.py` - The job listing scraper.
- `run_scraper.py` - Runs the scraper in a loop every few hours.

#### **🔑 Requirements:**
- Requires **Bright Data** credentials (**USERNAME** and **PASSWORD**). Need to store it in the `.env` file.
- The scraper runs automatically when `run_scraper.py` is executed.

### Data Preprocessing and Vector Database Setup
#### **🔍 Files (from the [preprocessing and pinecone](https://github.com/itamar-shashar/JobMatcher_AI/tree/main/preprocessing%20and%20pinecone) folder:**
- `analys_preprocess_and_vectordb.ipynb` - Jupyter Notebook for:
  - 🛠️ **Data cleaning** and **preprocessing** with **Apache Spark**.
  - 🔡 **Text normalization** and **feature engineering** (e.g., adding education level, filling missing values).
  - 🧩 **Chunking job descriptions** using **Semantic Graph Chunking**.
  - 📦 **Storing job listings in Pinecone Vector DB**.
- `semantic_graph_chunker.py` - Standalone script for **Semantic Graph Chunking**.

#### **⚙️ How to Run:**
- Best run on **Databricks** due to large-scale data processing.
- 📂 The dataset is not included in this repository. You need to either:
  1. **Obtain access** to the preprocessed parquet file.
  2. **Create your own parquet file** containing the following columns:
     - `company`, `title`, `location`, `job_type`, `salary`, `seniority`, `url`, `description`.

    Example for Dataset Structure:
<p align="center">
  <img src="https://github.com/user-attachments/assets/6176bff0-c2ad-4d6f-87ed-59beb4b99f36" alt="Example Dataset" style="width:80%;">>
</p>

- Set the **data directory, Pinecone API key, and index name** in the first cell of the notebook (need to copy from the .env file).
- ⏳ **Running this step takes time, but it is not necessary to use the app**, as the data must already exist in Pinecone.

---

## Notes & Future Improvements
- 🔑 The `.env` file containing API keys is **not included** in this repository and must be obtained separately.
- 📌 The preprocessing step is **not required** to run the app, as the job data is already indexed.
- ⚡ Future work will focus on **optimizing speed**, **improving prompt engineering**, and **scaling the system** to integrate with real-time job boards.

---

## License
This project is intended for educational and research purposes.

💬 For any inquiries, please open an issue or reach out to the contributors.

