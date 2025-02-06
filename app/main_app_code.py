import warnings
warnings.filterwarnings('ignore', category=UserWarning)

## Core
import os
from typing import List, Dict, Any
from collections import defaultdict

# File processing
from io import BytesIO
import PyPDF2
from docx import Document
import json

# AI/ML
import cohere
from pinecone import Pinecone as PineconeClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Utils
from rapidfuzz import fuzz
from dotenv import load_dotenv

# =============================================================================
# Global variables
# =============================================================================
load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

REFINEMENT_PROMPT = """You are an expert job search assistant. Your task is to analyze the user's query and CV to provide refined search parameters.

TASK 1: Create a refined search query that:
- Use the CV to create a more informative and robust query with more information for job search that improves vector search results.
- Maintains the original search intent
- Incorporates relevant experience, education and skills from the CV
- Adds important keywords from the CV that match the job search
- Keeps the query concise and focused


TASK 2: Extract specific search filters from both the query and CV, following these strict rules. **Always prefer the query over the CV if there is a conflict. Do not fabricate or infer details unless there is clear evidence in the provided texts.**

Filters:

- **location:**  
  * ONLY accept:
    1. U.S. locations in the EXACT format "City, ST" (e.g., "Boston, MA").
    2. The word "Remote".
  * SET TO null if:
    - The location is a non-U.S. location (e.g., "London, UK").
    - The state code is missing (e.g., "Chicago").
    - It is an international city (e.g., "Tokyo").
    - It includes a country name (e.g., "Paris, France").
    - It is not in the correct U.S. "City, ST" format (e.g., "New York, New York").

- **company:**  
  * Extract the company name only if clearly mentioned in the query (e.g., "Google", "Microsoft").  
  * SET TO null if:
    - No company is mentioned.
    - The reference to a company is ambiguous.
    - Only an industry is mentioned without a specific company.

- **job_title:**  
  * Extract the main role being sought (e.g., "Data Scientist").  
  * SET TO null if not clearly specified.

- **education_level:**  
  * Accept one of: ["high_school", "bachelor", "master", "phd", "None"].  
  * Use the value mentioned in the query if available.  
  * If not, infer it from the CV only if the evidence is strong.  
  * Otherwise, return null.

- **seniority_level:**  
  * Accept one of: ["Internship", "Entry level", "Associate", "Mid-senior level", "Senior level", "Director", "Executive", "None"].  
  * Choose the highest plausible seniority level based on the evidence from the query and CV, but do not make unrealistic choices.  
  * Use the query's information if available; if not, infer from the CV only when clearly supported.  
  * If there is no clear evidence, return null.

- **job_type:**  
  * Accept one of: ["full time", "part time", "contract", "temporary", "internship", "None"].  
  * Use the value from the query if available.  
  * If not, infer from the CV only when clearly supported by the text.  
  * If there is no clear evidence, return null.

User Query: "{query}"
CV Content: "{cv_text}"

Return ONLY a JSON object in this exact format (without any additional text):

{{
  "refined_query": "<refined query>",
  "filters": {{
    "location": "<location or null>",
    "company": "<company or null>",
    "title": "<job_title or null>",
    "education": "<education_level or null>",
    "seniority": "<seniority_level or null>",
    "job_type": "<job_type or null>"
  }}
}}

Remember: Only use information that is clearly supported by the user's query or CV. If the necessary details are not present, do not infer or make up information—return null for that field."""

GENERATOR_PROMPT = """You are an expert job search assistant. Focus ONLY on what's explicitly mentioned in the user's query: "{refined_query}" and their resume: 

### User Resume:
{resume}

Compare this query and resume with these job opportunities:

{job_texts}

Return ONLY a JSON array of objects, where each object MUST have:
- "job_number": the number of the job (1, 2, or 3) **(Only if the job is a match)**
- "summary": starts with "This position matches your search because", maximum 2 lines, only concrete matches

### **Requirements for Including a Job in the Results:**
A job **must be excluded** if **any** of the following conditions are met:
1. **Irrelevant job title** → The job title does **not match** the intent of the query.
2. **Mismatch in required education** → The user **does not meet** the job’s minimum education level.
3. **Insufficient years of experience** → The job **requires more experience** than the user has.
4. **Location mismatch** → If the query specifies a **location**, but the job is in a **different location**, **exclude it** (unless the job is remote).
5. **General mismatch** → The job **lacks strong alignment** with both the query and resume.

### **🔹 Summary Requirements**
- **Must start with:** `"This position matches your search because"`.
- **Only mention concrete matches** between **query, resume, and job description**.
- **Be factual**—no assumptions or vague statements.
- **Be at most 2 lines long**.


**Important:** If none of the jobs meet the criteria, return an **empty JSON array** (`[]`).
"""



# =============================================================================
# First, init all tools, vector DB and LLMs we will use
# =============================================================================
def init_llms_and_tools():
    cohere_client = cohere.Client(COHERE_API_KEY)
    embeddings_model = SentenceTransformer('intfloat/e5-base-v2')

    pinecone = PineconeClient(api_key=PINECONE_API_KEY)
    pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

    reranking_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranking_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return {
        'cohere_client': cohere_client,
        'embeddings_model': embeddings_model,
        'pinecone_index': pinecone_index,
        'reranking_tokenizer': reranking_tokenizer,
        'reranking_model': reranking_model,
    }


# =============================================================================
# Step 1: Process the user's CV
# =============================================================================
def _process_cv_document(uploaded_file=None):

    """Extract text from uploaded PDF or DOCX file"""
    if uploaded_file is None:
        return ""
        
    # Get the file type
    file_type = uploaded_file.type
    
    try:
        if file_type == 'application/pdf':
            # Read PDF
            pdf = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
                
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            # Read DOCX
            doc = Document(BytesIO(uploaded_file.read()))
            text = '\n'.join([para.text for para in doc.paragraphs])
            
        else:
            return None     
        return ' '.join(text.split())  # Basic cleaning
        
    except Exception as e:
        return None
        

# =============================================================================
# Step 2: Refine the query and extract filters
# =============================================================================
def _refine_query_with_cv(query, cv_text, cohere_client):

    prompt = REFINEMENT_PROMPT.format(
        query=query,
        cv_text=cv_text if cv_text else ""
    )

    response = cohere_client.generate(
        prompt=prompt,
        model="command-nightly",
        max_tokens=200,
        temperature=0.0,
        k=1
    )
    result = json.loads(response.generations[0].text)
        
    # Return tuple of (refined_query, filters)
    return result["refined_query"], result["filters"]


# =============================================================================
# Step 3: Search the Pincone vectorstore
# =============================================================================
def _vector_search(index, embeddings_model, refined_query, filters, top_k=50, score_threshold=0.8):
    """Two-stage search: content match first, then refine with title match"""

    # Build basic filter dictionary
    filter_dict = {key: {'$eq': filters[key]} for key in ["location", "education", "seniority"] if filters.get(key)}
    if filters.get("job_type"):
        filter_dict["job_type"] = {"$in": [filters["job_type"]]}

    # Stage 1: Content-based search using the refined query
    content_embedding = embeddings_model.encode(refined_query, normalize_embeddings=True).tolist()
    content_results = index.query(
        vector=content_embedding,
        filter=filter_dict,
        top_k=top_k * 2,  # Get more results to refine in the second stage
        include_metadata=True
    )

    # Extract only relevant information and apply initial score threshold
    content_matches = [{'id': item.id, 'metadata': item.metadata, 'score': item.score} for item in content_results.matches if item.score >= score_threshold]
    
    # # Apply initial score threshold
    # content_matches = [item for item in content_results if item.matches.score >= score_threshold]

    if not content_matches:
        return []  # No relevant content matches found, return empty list

    # Get job_ids of content matches
    matched_job_ids = [item['metadata']['job_id'] for item in content_matches]

    # Update the filter dictionary to include only the matched job IDs
    filter_dict["job_id"] = {"$in": matched_job_ids}

    # Stage 2: Title-based refinement on the filtered job IDs
    if filters.get("title"):
        title_query = f"Title: {filters['title']} role position job"
        title_embedding = embeddings_model.encode(title_query, normalize_embeddings=True).tolist()

        title_results = index.query(
            vector=title_embedding,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True
        )

        # Apply final score threshold
        return [
            {"id": item.id, "metadata": item.metadata, "score": item.score}
            for item in title_results.matches
            if item.score >= score_threshold
        ]

    # If no title filter, return the content-based results
    return [
        {"id": item.id, "metadata": item.metadata, "score": item.score}
        for item in content_matches
    ]


# =============================================================================
# Step 4: Re-rank the results using a cross-encoder
# =============================================================================
def _rerank_results(refined_query: str, vector_search_results: List[Dict], 
                   reranking_model, reranking_tokenizer, top_k=3):
    """
    Rerank results and return top_k documents with different job_ids.
    """
    # Store logits and documents
    logits_list = []
    documents_list = []

    # Loop through documents and collect logits
    for document in vector_search_results:
        context = document['metadata']['text']
        inputs = reranking_tokenizer.encode_plus(
            refined_query, context, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )

        # Get logits from the model
        with torch.no_grad():
            output = reranking_model(**inputs)
            logits = output.logits.item()

        logits_list.append(logits)
        documents_list.append(document)

    # Normalize scores using min-max scaling
    logits_tensor = torch.tensor(logits_list)
    min_logit, max_logit = logits_tensor.min(), logits_tensor.max()
    scores = ((logits_tensor - min_logit) / (max_logit - min_logit)).tolist()

    # Combine scores with documents
    scored_results = [
        {**doc, "reranking_score": score} 
        for doc, score in zip(documents_list, scores)
    ]

    # Sort by score
    scored_results.sort(key=lambda x: x["reranking_score"], reverse=True)

    # Get top k different jobs
    seen_job_ids = set()
    unique_job_results = []
    
    for result in scored_results:
        job_id = result['metadata']['job_id']
        if job_id not in seen_job_ids:
            seen_job_ids.add(job_id)
            unique_job_results.append(result)
            if len(unique_job_results) == top_k:
                break

    return unique_job_results


# =============================================================================
# Step 5: Aggregate job descriptions for top results
# =============================================================================
def _aggregate_jobs(reranked_documents: List[Dict], vectorstore_index) -> List[Dict]:
    """
    Reassembles complete job descriptions from chunks efficiently.
    
    Args:
        reranked_documents: List of reranked document chunks
        vectorstore_index: Pinecone index instance
    
    Returns:
        List of complete job descriptions with metadata, sorted by reranking score
    """
    if not reranked_documents:
        return []
    
    # Extract unique job IDs
    job_ids = list({doc["metadata"]["job_id"] for doc in reranked_documents})  

    # Generate the modified job IDs 
    items_ids = [f"{int(job_id)}_{i}" for job_id in job_ids for i in range(10)] # There are no more than 10 chunks per job

    # Search the index by items IDs
    fetched_results = vectorstore_index.fetch(ids=items_ids) 
    
    results_metadatas = [item.metadata for item in fetched_results.vectors.values()]

    job_chunks = defaultdict(list)
    for chunk in results_metadatas:
        job_chunks[chunk['job_id']].append(chunk)

    
    # Process each job (list comprehension for efficiency)
    reassembled_jobs = []
    for _, chunks_list in job_chunks.items():
        # Sort chunks once
        sorted_chunks = sorted(chunks_list, key=lambda chunk: chunk["chunk_index"])
        
        # Extract relevant metadata
        metadata = {key: sorted_chunks[0][key] for key in ['company', 'title', 'location', 'url']}
        
        # Join texts efficiently
        full_description = " ".join(
            chunk["text"].split("\n", 1)[1] 
            for chunk in sorted_chunks
        ).strip()

        reassembled_jobs.append({
            "job_description": full_description,
            "metadata": metadata
        })

    return reassembled_jobs


# =============================================================================
# Step 6: Creare a final respons generated by LLM
# =============================================================================

def _create_job_suggestions(refined_query, resume, aggregated_jobs, cohere_client):
    """
    Create personalized job suggestions with selected original metadata.
    """
    # Build job descriptions string directly
    job_texts = ""
    for i, job in enumerate(aggregated_jobs, start=1):
        job_texts += (
            f"Job {i}:\n"
            f"Title: {job['metadata']['title']}\n"
            f"Company: {job['metadata']['company']}\n"
            f"Location: {job['metadata']['location']}\n"
            f"Description: {job['job_description']}...\n\n"
        )

    prompt = GENERATOR_PROMPT.format(refined_query=refined_query, resume=resume, job_texts=job_texts )

    response = cohere_client.generate(
        prompt=prompt,
        model="command-nightly",
        max_tokens=300,
        temperature=0.2,
        k=1,
    )

    try:
        summaries = json.loads(response.generations[0].text)
        
        # Create a mapping of job number to summary
        summary_map = {
            item['job_number']: item['summary'] 
            for item in summaries
        }
        
        # Combine summaries with selected metadata
        results = []        
        for i, job in enumerate(aggregated_jobs, start=1):
            results.append({
                "summarization": summary_map.get(i, "No summary available"),  
                "metadata": job['metadata']
            })
        
        return results
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Cohere: {response.generations[0].text}") from e


# =============================================================================
# Step 6: Overall Process Function
# =============================================================================


def process_search(llms_and_tools, query, cv_file=None, progress_callback=None):
    """
    Main search function that:

    """

    # Step 1: Process the user's CV
    if progress_callback:
        progress_callback("🔍 Analyzing your requirements...", 0.0)
    cv_text = _process_cv_document(cv_file)
    
    # Step 2: Refine the query and extract filters.
    if progress_callback:
        progress_callback("📝 Refining search criteria...", 0.12)
    refined_query, filters = _refine_query_with_cv(query=query, 
                                                   cv_text=cv_text, 
                                                   cohere_client=llms_and_tools['cohere_client']
                                                   )
    
    # Step 3: Vector search via Pinecone.
    if progress_callback:
        progress_callback("🔎 Searching job database...", 0.3)
    search_results = _vector_search(index=llms_and_tools['pinecone_index'], 
                                    embeddings_model=llms_and_tools['embeddings_model'], 
                                    refined_query=refined_query, 
                                    filters=filters
                                    )
    if not search_results:
        return []
    
    # Step 4: Re-rank the results using a cross-encoder.
    if progress_callback:
        progress_callback("🎯 Finding best matches...", 0.5)
    reranked_results = _rerank_results(refined_query=refined_query, 
                                       vector_search_results=search_results, 
                                       reranking_model=llms_and_tools['reranking_model'], 
                                       reranking_tokenizer=llms_and_tools['reranking_tokenizer'] 
                                       )
    if not reranked_results:
        return []
    
    # Step 5: Aggregate job chunks by job_id.
    if progress_callback:
        progress_callback("⚡ Calculating relevance scores...", 0.67)
    aggregated_jobs = _aggregate_jobs(reranked_documents=reranked_results, 
                                      vectorstore_index=llms_and_tools['pinecone_index']
                                      )
    
    # Step 6: Create a final response generated by LLM.
    if progress_callback:
        progress_callback("📋 Preparing personalized results...", 85)
    job_suggestions = _create_job_suggestions(refined_query=refined_query,
                                              resume=cv_text,
                                               aggregated_jobs=aggregated_jobs, 
                                               cohere_client=llms_and_tools['cohere_client']
                                               )
    
    if progress_callback:
        progress_callback("✅ Complete!", 1.0)

    # Return the top job listings for the user.
    return job_suggestions
