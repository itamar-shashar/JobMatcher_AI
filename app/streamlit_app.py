import streamlit as st
import time
from main_app_code import process_search, init_llms_and_tools

st.set_page_config(page_title="AI Job Matcher", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background-color: #0D1117;
  color: #FFFFFF;
}

[data-testid="stTextArea"] label, [data-testid="stFileUploader"] label {
  color: white !important;
}

.stTextArea textarea {
  background-color: #161B22;
  color: #FFFFFF;
  border: 1px solid #30363D;
  font-size: 16px;
  padding: 15px;
}

.stButton>button {
  background-color: #238636;
  color: white;
  border-radius: 6px;
  padding: 12px 24px;
  font-weight: 600;
  width: 100%;
  border: none;
  margin-top: 15px;
  font-size: 16px;
}

.stButton>button:hover {
  background-color: #2EA043;
  transform: translateY(-2px);
  transition: all 0.2s ease;
}

.upload-box {
  background-color: #161B22;
  border: 2px dashed #30363D;
  border-radius: 8px;
  padding: 25px;
  text-align: center;
  margin: 20px 0;
}

.progress-stage {
  color: #8B949E;
  font-size: 16px;
  margin: 15px 0;
  padding: 10px;
  border-radius: 6px;
}

.stage-active {
  color: #FFFFFF;
  font-weight: 600;
  background-color: #1F2937;
}

.job-card {
  background-color: #161B22;
  padding: 25px;
  border-radius: 12px;
  margin: 20px 0;
  border: 1px solid #30363D;
  transition: transform 0.2s ease;
}

.job-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.job-title {
  color: #58A6FF;
  font-size: 22px;
  font-weight: 600;
  margin-bottom: 12px;
}

.job-company {
  color: #8B949E;
  font-size: 16px;
  margin-bottom: 15px;
}

.job-summary {
  margin: 20px 0;
  line-height: 1.6;
}

.view-job-btn {
  display: inline-block;
  background-color: #238636;
  color: #FFFFFF !important;
  padding: 10px 20px;
  border-radius: 6px;
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.view-job-btn:hover {
  background-color: #2EA043;
  transform: translateY(-2px);
  color: #FFFFFF !important;
}

.stProgress > div > div {
  width: 100% !important;
}

h1, h2, h3 {
  font-weight: 600;
  margin-bottom: 20px;
}
            
.no-results-card {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    margin: 40px 0;
    animation: fadeIn 0.5s ease-in-out;
}

.no-results-emoji {
    font-size: 48px;
    margin-bottom: 20px;
}

.no-results-title {
    color: #58A6FF;
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 15px;
}

.no-results-message {
    color: #8B949E;
    font-size: 20px;
    line-height: 1.6;
    margin-bottom: 20px;
}

.no-results-tips {
    background-color: #1F2937;
    border-radius: 8px;
    padding: 15px;
    margin-top: 20px;
    text-align: left;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2])

with left_col:
   st.markdown("# JobMatcher AI") 
   st.markdown("#### Find your perfect job match using AI")
   
   query = st.text_area(
       "What kind of job are you looking for?",
       placeholder="Describe your ideal role, required skills, and preferences...", 
       height=200
   )
   
   uploaded_cv = st.file_uploader(
       "Upload your CV (optional)",
       type=["pdf", "docx"],
       help="Upload your CV to improve job matching accuracy"
   )
   
   search_button = st.button("Find Matching Jobs 🎯")

# Initialize tools
llms_and_tools = init_llms_and_tools()

with right_col:
    if search_button and query:
        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(stage, value):
            status.markdown(
                f'<div class="progress-stage stage-active">{stage}</div>',
                unsafe_allow_html=True
            )
            progress_bar.progress(value)

        # Process search with progress updates
        results = process_search(
            llms_and_tools=llms_and_tools, 
            query=query,
            cv_file=uploaded_cv,
            progress_callback=update_progress
        )

        progress_bar.empty()
        status.empty()
        
        if not results:
          st.markdown("""
              <div class="no-results-card">
                  <div class="no-results-emoji">😕</div>
                  <div class="no-results-title">Sorry... We Could't Find A Match</div>
                  <div class="no-results-message">
                      But don't worry, let's try to improve your search!
                  </div>
                  <div class="no-results-tips">
                      Try these tips:
                      <ul>
                          <li>Be more specific about the role you're looking for</li>
                          <li>Include key skills or technologies you know</li>
                          <li>Mention your preferred location or if you're open to remote work</li>
                          <li>Specify your experience level</li>
                      </ul>
                  </div>
              </div>
          """, unsafe_allow_html=True)

        else:
            # Display results
            st.markdown("## Best Matches For You")
            for res in results:
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">{res["metadata"]["title"]}</div>
                        <div class="job-company">
                            🏢 {res["metadata"]["company"]} | 📍 {res["metadata"]["location"]}
                        </div>
                        <div class="job-summary">{res["summarization"]}</div>
                        <a class="view-job-btn" href="{res["metadata"]["url"]}" target="_blank">
                            View Full Job Details →
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
