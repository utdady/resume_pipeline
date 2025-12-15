"""Streamlit web UI for Resume Matching Pipeline."""
import copy
import io
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import yaml

from baseline import score_resume, load_config
from parser import extract_text


# Page configuration
st.set_page_config(
    page_title="Resume Matching Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def save_uploaded_files(uploaded_files: List, temp_dir: Path) -> List[Path]:
    """Save uploaded files to temporary directory."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def process_job_description(jd_text: str, job_title: Optional[str] = None, 
                           req_id: Optional[str] = None) -> Path:
    """Process job description and generate config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_path = Path(f.name)
    
    generate_jd_meta(jd_text, config_path, job_title=job_title, requisition_id=req_id)
    return config_path


def main():
    """Main Streamlit app."""
    st.markdown('<p class="main-header">📄 Resume Matching Pipeline</p>', 
                unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Home", "Score Resumes", "View Results"])
    
    if page == "Home":
        show_home_page()
    elif page == "Score Resumes":
        show_scoring_page()
    elif page == "View Results":
        show_results_page()


def show_home_page():
    """Display home page with instructions."""
    st.header("Welcome to Resume Matching Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 How It Works")
        st.markdown("""
        1. **Enter Job Description** - Paste or upload a job description
        2. **Upload Resumes** - Upload multiple resume files (PDF, DOCX, TXT)
        3. **Generate Scores** - System automatically:
           - Analyzes job requirements
           - Extracts must-haves and nice-to-haves
           - Scores each resume
        4. **View Results** - See ranked candidates with detailed breakdowns
        """)
    
    with col2:
        st.subheader("🎯 Features")
        st.markdown("""
        - **Auto Job Analysis** - Extracts requirements from any JD
        - **Domain Detection** - Identifies job domain automatically
        - **Smart Scoring** - Rule-based matching with weighted components
        - **Detailed Reports** - See what's missing and what's strong
        - **Export Results** - Download CSV for further analysis
        """)
    
    st.divider()
    
    st.subheader("📊 Scoring Components")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Must-Have Coverage", "45%", "Critical requirements")
    with col2:
        st.metric("Skill Overlap", "25%", "Nice-to-have skills")
    with col3:
        st.metric("Title Similarity", "15%", "Role alignment")
    with col4:
        st.metric("Years Experience", "15%", "Experience match")
    
    st.info("💡 **Tip**: Go to 'Score Resumes' to start analyzing candidates!")


def show_scoring_page():
    """Display the main scoring interface."""
    st.header("📊 Score Resumes Against Job Description")
    
    # Step 1: Job Description Upload & Analysis
    st.subheader("Step 1: Upload & Analyze Job Description")
    st.info("📄 Upload a job description (TXT, DOCX, or PDF). The system will parse it, extract requirements, and generate the scoring YAML automatically.")
    
    uploaded_jd = st.file_uploader(
        "Upload Job Description File",
        type=["txt", "docx", "pdf"],
        help="Upload a job description in TXT, DOCX, or PDF format. The system will automatically parse and analyze it.",
        label_visibility="visible"
    )
    
    jd_text = ""
    job_title = None
    req_id = None
    
    with st.expander("📝 Or paste job description text instead", expanded=False):
        jd_text_paste = st.text_area(
            "Job Description Text",
            height=200,
            placeholder="Paste the job description here if you prefer text input..."
        )
        if jd_text_paste:
            jd_text = jd_text_paste
    
    if uploaded_jd:
        try:
            status_placeholder = st.empty()
            status_placeholder.info(f"📄 Processing {uploaded_jd.name}...")
            
            if uploaded_jd.type == "text/plain" or uploaded_jd.name.endswith(".txt"):
                jd_text = uploaded_jd.read().decode("utf-8", errors="ignore")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_jd.name).suffix) as tmp:
                    tmp.write(uploaded_jd.getbuffer())
                    tmp_path = Path(tmp.name)
                    jd_text = extract_text(tmp_path)
            
            status_placeholder.success(f"✅ Successfully parsed {uploaded_jd.name} ({len(jd_text)} characters)")
            
            with st.expander("👁️ Preview extracted job description text", expanded=False):
                st.text(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            jd_text = ""
    
    # Optional metadata overrides
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input(
            "Job Title (optional)",
            placeholder="e.g., Financial Analyst",
            help="Leave empty to auto-detect from job description"
        )
    with col2:
        req_id = st.text_input(
            "Requisition ID (optional)",
            placeholder="e.g., FA-2024-001",
            help="Optional identifier for this job posting"
        )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🔍 Analyze Job Description", use_container_width=True)
    with col2:
        use_llm = st.checkbox("🧠 AI", value=True, help="Use Ollama AI for smarter analysis")
    
    if analyze_button:
        if not jd_text.strip():
            st.error("❌ Please upload or paste a job description before analyzing.")
        else:
            try:
                from jd_analyzer_llm import analyze_jd_hybrid, check_ollama_available
                
                if use_llm and not check_ollama_available():
                    st.warning("⚠️ Ollama not running. Using rule-based analysis instead.")
                    st.info("To use AI: Install Ollama and run 'ollama serve' (see setup_ollama.md)")
                    use_llm = False
                
                with st.spinner("Analyzing job description and generating YAML config..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp_yaml:
                        config_path = Path(tmp_yaml.name)
                    
                    config = analyze_jd_hybrid(
                        jd_text,
                        config_path,
                        use_llm=use_llm,
                        job_title=job_title,
                        requisition_id=req_id
                    )
                    
                    st.session_state["base_config"] = config
                    st.session_state["custom_config"] = copy.deepcopy(config)
                    st.session_state["config_yaml"] = yaml.dump(config, sort_keys=False, allow_unicode=True)
                    st.session_state["job_title"] = config.get("job", {}).get("title", "Job Position")
                
                if "Ollama" in config.get("notes", ""):
                    st.success("✅ Job description analyzed with Ollama AI! Review and customize below.")
                else:
                    st.success("✅ Job description analyzed with rule-based system! Review and customize below.")
            except Exception as e:
                st.error(f"❌ Error analyzing job description: {e}")
                st.exception(e)
    
    # Display extracted requirements and customization options
    if "custom_config" in st.session_state:
        config = st.session_state["custom_config"]
        domain = config.get("job", {}).get("domain", "general")
        
        with st.expander("📋 View & Download Current YAML Configuration", expanded=True):
            st.write(f"**Job Title:** {config.get('job', {}).get('title', 'N/A')}")
            st.write(f"**Domain:** {domain}")
            st.write(f"**Requisition ID:** {config.get('job', {}).get('requisition_id', 'N/A')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Must-Have Requirements:**")
                for mh in config.get("must_haves", []):
                    st.write(f"  • {mh}")
            with col2:
                st.write("**Nice-to-Haves:**")
                for nh in config.get("nice_to_haves", []):
                    st.write(f"  • {nh}")
            
            exp = config.get("experience", {})
            st.write(f"**Experience Required:** {exp.get('min_years', 'N/A')} - {exp.get('max_years', 'N/A')} years (preferred: {exp.get('preferred_years', 'N/A')})")
            
            yaml_content = yaml.dump(config, sort_keys=False, allow_unicode=True)
            st.download_button(
                label="📥 Download Current jd_meta.yaml",
                data=yaml_content,
                file_name="jd_meta.yaml",
                mime="text/yaml",
                help="Download the current YAML configuration (including customizations)"
            )
        
        with st.expander("🔧 Customize Requirements & Weights", expanded=False):
            with st.form("custom_config_form", clear_on_submit=False):
                must_text = st.text_area(
                    "Must-Have Requirements (one per line)",
                    value="\n".join(config.get("must_haves", [])),
                    height=150
                )
                nice_text = st.text_area(
                    "Nice-to-Have Skills (one per line)",
                    value="\n".join(config.get("nice_to_haves", [])),
                    height=150
                )
                title_text = st.text_area(
                    "Title Keywords (one per line)",
                    value="\n".join(config.get("title_keywords", [])),
                    height=100
                )
                
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                with exp_col1:
                    min_years = st.number_input(
                        "Min Years Experience",
                        value=int(config.get("experience", {}).get("min_years", 0)),
                        step=1,
                        min_value=0
                    )
                with exp_col2:
                    preferred_years = st.number_input(
                        "Preferred Years",
                        value=int(config.get("experience", {}).get("preferred_years", max(min_years, 0))),
                        step=1,
                        min_value=0
                    )
                with exp_col3:
                    max_years = st.number_input(
                        "Max Years Experience",
                        value=int(config.get("experience", {}).get("max_years", max(preferred_years, min_years))),
                        step=1,
                        min_value=preferred_years if preferred_years else min_years
                    )
                
                st.write("**Adjust Scoring Weights (will normalize to 100%)**")
                weight_col1, weight_col2, weight_col3, weight_col4 = st.columns(4)
                with weight_col1:
                    weight_must = st.number_input(
                        "Must-Haves",
                        value=float(config.get("weights", {}).get("must_have_coverage", 0.45)),
                        step=0.05,
                        min_value=0.0,
                        format="%.2f"
                    )
                with weight_col2:
                    weight_skill = st.number_input(
                        "Skill Overlap",
                        value=float(config.get("weights", {}).get("skill_overlap", 0.25)),
                        step=0.05,
                        min_value=0.0,
                        format="%.2f"
                    )
                with weight_col3:
                    weight_title = st.number_input(
                        "Title Similarity",
                        value=float(config.get("weights", {}).get("title_similarity", 0.15)),
                        step=0.05,
                        min_value=0.0,
                        format="%.2f"
                    )
                with weight_col4:
                    weight_years = st.number_input(
                        "Years Experience",
                        value=float(config.get("weights", {}).get("years_exp", 0.15)),
                        step=0.05,
                        min_value=0.0,
                        format="%.2f"
                    )
                
                customize_submitted = st.form_submit_button("Apply Customizations")
                
                if customize_submitted:
                    updated_config = copy.deepcopy(config)
                    updated_config["must_haves"] = [line.strip() for line in must_text.splitlines() if line.strip()]
                    updated_config["nice_to_haves"] = [line.strip() for line in nice_text.splitlines() if line.strip()]
                    updated_config["title_keywords"] = [line.strip() for line in title_text.splitlines() if line.strip()]
                    updated_config["experience"] = {
                        "min_years": int(min_years),
                        "max_years": int(max_years),
                        "preferred_years": int(preferred_years)
                    }
                    
                    weights = {
                        "must_have_coverage": weight_must,
                        "skill_overlap": weight_skill,
                        "title_similarity": weight_title,
                        "years_exp": weight_years
                    }
                    total_weight = sum(weights.values())
                    if total_weight <= 0:
                        st.error("❌ Total weight must be greater than zero.")
                    else:
                        normalized_weights = {k: v / total_weight for k, v in weights.items()}
                        updated_config["weights"] = normalized_weights
                        st.session_state["custom_config"] = updated_config
                        st.session_state["config_yaml"] = yaml.dump(updated_config, sort_keys=False, allow_unicode=True)
                        st.success("✅ Customizations applied! Future scores will use the updated configuration.")
    else:
        st.info("Upload and analyze a job description to unlock customization and scoring.")
    
    st.divider()
    
    # Step 2: Resume Upload
    st.subheader("Step 2: Upload Resumes")
    
    uploaded_resumes = st.file_uploader(
        "Upload Resume Files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or more resume files. Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_resumes:
        st.success(f"✅ {len(uploaded_resumes)} resume file(s) uploaded")
        st.write("**Uploaded files:**")
        for file in uploaded_resumes:
            st.write(f"- {file.name}")
    
    st.divider()
    
    # Step 3: Score resumes using the customized configuration
    st.subheader("Step 3: Generate Scores")
    
    if st.button("🚀 Score Resumes", type="primary", use_container_width=True):
        if "custom_config" not in st.session_state:
            st.error("❌ Please analyze a job description before scoring resumes.")
            return
        
        if not uploaded_resumes:
            st.error("❌ Please upload at least one resume file!")
            return
        
        config = st.session_state["custom_config"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("💾 Saving uploaded resumes...")
            progress_bar.progress(10)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                resume_paths = save_uploaded_files(uploaded_resumes, temp_path)
                
                status_text.text(f"📊 Scoring {len(resume_paths)} resume(s)...")
                progress_bar.progress(40)
                
                results = []
                for i, resume_path in enumerate(resume_paths):
                    progress = 40 + int((i + 1) / len(resume_paths) * 50)
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Scoring {resume_path.name}... ({i+1}/{len(resume_paths)})")
                    
                    result = score_resume(resume_path, config)
                    results.append(result)
            
            progress_bar.progress(95)
            status_text.text("📈 Generating results table...")
            
            df = pd.DataFrame(results)
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            
            st.session_state["results_df"] = df
            st.session_state["config"] = config
            st.session_state["job_title"] = config.get("job", {}).get("title", "Job Position")
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            st.success(f"🎉 Successfully scored {len(results)} resume(s)!")
            st.balloons()
            
            show_results_summary(df)
            st.subheader("📊 Results Table")
            show_results_table(df)
        except Exception as e:
            st.error(f"❌ Error scoring resumes: {e}")
            st.exception(e)


def show_results_summary(df: pd.DataFrame):
    """Display summary statistics."""
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", len(df))
    
    with col2:
        advance_count = len(df[(df['score'] >= 0.72) & (df['must_haves_present'] == df['must_haves_total'])])
        st.metric("Advance Qualified", advance_count)
    
    with col3:
        review_count = len(df[df['recommendation'] == 'Review'])
        st.metric("Review", review_count)
    
    with col4:
        reject_count = len(df[df['recommendation'] == 'Reject'])
        st.metric("Reject", reject_count)
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Score Distribution:**")
        score_ranges = {
            "0.8 - 1.0": len(df[df['score'] >= 0.8]),
            "0.7 - 0.8": len(df[(df['score'] >= 0.7) & (df['score'] < 0.8)]),
            "0.5 - 0.7": len(df[(df['score'] >= 0.5) & (df['score'] < 0.7)]),
            "< 0.5": len(df[df['score'] < 0.5])
        }
        for range_name, count in score_ranges.items():
            st.write(f"- {range_name}: {count} candidates")
    
    with col2:
        st.write("**Top 5 Candidates:**")
        top5 = df.head(5)
        for idx, row in top5.iterrows():
            st.write(f"{idx+1}. {row['file']} - Score: {row['score']:.3f} ({row['recommendation']})")


def show_results_table(df: pd.DataFrame):
    """Display interactive results table."""
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.01)
    
    with col2:
        recommendation_filter = st.selectbox(
            "Recommendation",
            ["All", "Advance", "Review", "Reject"]
        )
    
    with col3:
        min_must_haves = st.slider("Min Must-Haves", 0, int(df['must_haves_total'].iloc[0]) if len(df) > 0 else 10, 0)
    
    # Apply filters
    filtered_df = df[df['score'] >= min_score]
    if recommendation_filter != "All":
        filtered_df = filtered_df[filtered_df['recommendation'] == recommendation_filter]
    filtered_df = filtered_df[filtered_df['must_haves_present'] >= min_must_haves]
    
    st.write(f"**Showing {len(filtered_df)} of {len(df)} candidates**")
    
    # Display table
    display_columns = ['file', 'score', 'recommendation', 'years_found', 
                      'must_haves_present', 'must_haves_total', 'explanation']
    
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="resume_scores.csv",
        mime="text/csv"
    )


def show_results_page():
    """Display saved results if available."""
    st.header("📊 View Results")
    
    if 'results_df' not in st.session_state:
        st.info("👈 Go to 'Score Resumes' to generate results first!")
        return
    
    df = st.session_state['results_df']
    config = st.session_state.get('config', {})
    job_title = st.session_state.get('job_title', 'Job Position')
    
    st.subheader(f"Results for: {job_title}")
    
    show_results_summary(df)
    show_results_table(df)
    
    # Detailed view
    st.subheader("🔍 Detailed Candidate View")
    selected_file = st.selectbox("Select a candidate to view details:", df['file'].tolist())
    
    if selected_file:
        candidate = df[df['file'] == selected_file].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Score Breakdown:**")
            st.metric("Overall Score", f"{candidate['score']:.3f}")
            st.metric("Recommendation", candidate['recommendation'])
            st.metric("Years Found", f"{candidate['years_found']:.1f}")
            st.metric("Must-Haves", f"{candidate['must_haves_present']}/{candidate['must_haves_total']}")
        
        with col2:
            st.write("**Component Scores:**")
            st.metric("Must-Have Coverage", f"{candidate['must_have_coverage']:.1%}")
            st.metric("Skill Overlap", f"{candidate['skill_overlap']:.1%}")
            st.metric("Title Similarity", f"{candidate['title_similarity']:.1%}")
            st.metric("Years Experience", f"{candidate['years_score']:.1%}")
        
        st.write("**Explanation:**")
        st.info(candidate['explanation'])


if __name__ == "__main__":
    main()
