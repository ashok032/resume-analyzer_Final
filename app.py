import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from supabase import create_client, Client
from spacy.matcher import Matcher

# --- Load secrets from Streamlit's secrets management ---
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    SMTP_EMAIL = st.secrets["email"]["smtp_email"]
    SMTP_PASSWORD = st.secrets["email"]["smtp_password"]
except KeyError:
    st.error("Supabase or Email secrets are not configured correctly in Streamlit Cloud. Please check your settings.")
    st.stop()

# --- Supabase & Email Configuration ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Could not connect to Supabase. Please check your URL and Key. Error: {e}")
    st.stop()
    
# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.info("Downloading spaCy model... Please wait.")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# -------------------- Helpers --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- ROBUST RESUME PARSING FUNCTIONS ---
def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def extract_email(text):
    """Extracts email from text using regex."""
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Not found"

# --- CORRECTED HYBRID SKILL EXTRACTION ---
def extract_keywords(text):
    """
    Extracts skills using the proven hybrid approach for the best balance of precision and discovery.
    """
    original_text = text
    
    # --- FIX: Clean text to handle punctuation that can separate skill words ---
    # This turns "(Spring Boot)" or "Spring Boot," into "Spring Boot " for easier matching.
    cleaned_text_for_spacy = re.sub(r'[(),|]', ' ', original_text)
    
    doc = nlp(cleaned_text_for_spacy.lower()) # Use cleaned text for spaCy processing
    matcher = Matcher(nlp.vocab)
    keywords = set()
    matched_tokens = set()

    IGNORE_WORDS = {
        'experience', 'skill', 'skills', 'profile', 'summary', 'education', 'project', 
        'projects', 'internship', 'internships', 'work', 'role', 'contact', 'email', 
        'phone', 'address', 'linkedin', 'github', 'name', 'date', 'month', 'year', 
        'company', 'university', 'college', 'gpa', 'description', 'responsibility', 
        'responsibilities', 'objective', 'team', 'inc', 'ltd'
    }

    # --- Step 1: High-Precision Matcher for common, unambiguous skills ---
    patterns = {
        "java": [[{"LOWER": "java"}]], "sql": [[{"LOWER": "sql"}]], "python": [[{"LOWER": "python"}]],
        "react": [[{"LOWER": "react"}]], "spring boot": [[{"LOWER": "spring"}, {"LOWER": "boot"}]], 
        "rest api": [[{"LOWER": "rest"}, {"LOWER": "api"}]], "machine learning": [[{"LOWER": "machine"}, {"LOWER": "learning"}]], 
        "core java": [[{"LOWER": "core"}, {"LOWER": "java"}]], "microservices": [[{"LOWER": "microservices"}]], 
        "node.js": [[{"LOWER": "node"}, {"IS_PUNCT": True}, {"LOWER": "js"}]]
    }

    for skill, pattern in patterns.items():
        matcher.add(skill, pattern)

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        span_text = span.text
        keywords.add(span_text)
        
        if span_text == "core java": keywords.add("java")
        elif span_text == "spring boot": keywords.add("spring")
        elif span_text == "rest api": keywords.add("api"); keywords.add("rest")
        elif span_text == "node.js": keywords.add("node")

        for i in range(start, end):
            matched_tokens.add(i)

    # --- Step 2: General Noun Chunk Extraction to discover ANY other skill ---
    for chunk in doc.noun_chunks:
        if chunk.start not in matched_tokens and chunk.end - 1 not in matched_tokens:
            clean_chunk = chunk.lemma_.strip()
            if len(clean_chunk) > 2 and clean_chunk not in IGNORE_WORDS and not any(word in IGNORE_WORDS for word in clean_chunk.split()):
                keywords.add(clean_chunk)

    # --- Step 3: Extract Single-Word PROPN and NOUN skills (as a fallback) ---
    for token in doc:
        if token.i not in matched_tokens and token.pos_ in ('PROPN', 'NOUN'):
            lemma = token.lemma_.strip()
            if len(lemma) > 1 and not token.is_stop and lemma not in IGNORE_WORDS:
                keywords.add(lemma)
                
    # --- Step 4: Use Regular Expressions for Specific Formats ---
    special_formats = re.findall(r'\b[A-Z]\+\+|\b[A-Z]#|\b\.NET\b', original_text)
    for skill in special_formats:
        keywords.add(skill.lower())

    return list(keywords)
    
def match_resume_to_job(resume_keywords, job_skills):
    resume_set = set(k.lower() for k in resume_keywords)
    job_set = set(s.lower() for s in job_skills)
    matched_skills = list(resume_set.intersection(job_set))
    missing_skills = list(job_set.difference(resume_set))
    score = (len(matched_skills) / len(job_set)) * 100 if job_set else 100
    return matched_skills, missing_skills, round(score, 2)

def send_email(to_email, subject, body):
    """Sends a simple plain text email."""
    if not to_email or to_email == "Not found":
        st.error("Email is blank, cannot send email.")
        return False
        
    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        st.success(f"Email sent to {to_email}")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# -------------------- Supabase Data Functions --------------------

@st.cache_data(ttl=600)
def load_jobs_from_db():
    response = supabase.table('jobs').select('id, title, company, skills, description').execute()
    return pd.DataFrame(response.data) if response.data else pd.DataFrame(columns=['id', 'title', 'company', 'skills', 'description'])

def load_all_applications():
    final_cols = ["id", "logged_in_username", "email", "candidate_name", "role", "company", "match_score", "current_phase", "status"]
    try:
        response = supabase.table('applications').select(
            'id, match_score, phase, status, candidate_name, candidate_email, users(email), jobs(title, company)'
        ).execute()
    except Exception as e:
        st.error(f"Failed to load application data: {e}")
        return pd.DataFrame(columns=final_cols)
    if not response.data:
        return pd.DataFrame(columns=final_cols)
    df = pd.json_normalize(response.data)
    df.rename(columns={
        'candidate_email': 'email', 'phase': 'current_phase',
        'users.email': 'logged_in_username', 'jobs.title': 'role', 'jobs.company': 'company'
    }, inplace=True)
    for col in final_cols:
        if col not in df.columns: df[col] = None
    return df[final_cols]

# -------------------- UI & Logic --------------------

def login_register_ui():
    st.title("AI Resume Analyzer")
    tabs = st.tabs(["Login", "Register"])
    with tabs[0]:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            hashed = hash_password(password)
            user = supabase.table('users').select('*').eq('email', email).eq('password_hash', hashed).execute()
            if user.data:
                st.session_state.update({"username": user.data[0]['email'], "role": user.data[0]['role'], "user_id": user.data[0]['id']})
                st.success("Login successful!"); st.rerun()
            else:
                st.error("Invalid credentials.")
        
        with st.expander("Forgot Password?"):
            with st.form("reset_password_form", clear_on_submit=False):
                st.write("Reset your password by entering your email and a new password.")
                email_reset = st.text_input("Enter your registered Email", key="email_reset")
                new_password_reset = st.text_input("Enter New Password", type="password", key="new_pass_reset")
                confirm_password_reset = st.text_input("Confirm New Password", type="password", key="confirm_pass_reset")
                
                submitted_reset = st.form_submit_button("Reset Password")
                
                if submitted_reset:
                    if not email_reset or not new_password_reset or not confirm_password_reset:
                        st.warning("Please fill in all fields.")
                    elif new_password_reset != confirm_password_reset:
                        st.error("Passwords do not match. Please try again.")
                    else:
                        user_check = supabase.table('users').select('id').eq('email', email_reset).execute()
                        if not user_check.data:
                            st.error("This email is not registered. Please check the email address.")
                        else:
                            try:
                                new_hashed_password = hash_password(new_password_reset)
                                supabase.table('users').update({'password_hash': new_hashed_password}).eq('email', email_reset).execute()
                                st.success("Password reset successfully! You can now log in with your new password.")
                            except Exception as e:
                                st.error(f"An error occurred while resetting the password: {e}")

    with tabs[1]:
        new_email = st.text_input("Email", key="reg_user")
        new_password = st.text_input("Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            if supabase.table('users').select('id').eq('email', new_email).execute().data:
                st.warning("Email already registered.")
            else:
                new_user = {"email": new_email, "password_hash": hash_password(new_password), "role": role}
                if supabase.table('users').insert(new_user).execute().data:
                    st.success("Registered successfully! Please login.")
                else:
                    st.error("Registration failed.")

def user_view():
    st.header("Candidate Dashboard")
    df_all = load_all_applications()
    my_apps = df_all[df_all["logged_in_username"] == st.session_state["username"]]
    st.subheader("My Applications")
    
    if not my_apps.empty:
        st.dataframe(my_apps[['company', 'role', 'match_score', 'current_phase', 'status']])
    else:
        st.info("You have not submitted any applications yet.")

    jobs_df = load_jobs_from_db()
    if jobs_df.empty:
        st.warning("No jobs are currently available."); return
        
    job_role = st.selectbox("Select a Job Role", options=jobs_df['title'].unique())
    company_options = jobs_df[jobs_df['title'] == job_role]['company'].unique()
    company = st.selectbox("Select Company", options=company_options)

    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
        
        candidate_email = extract_email(text)
        
        filtered_job_df = jobs_df[(jobs_df['title'] == job_role) & (jobs_df['company'] == company)]
        
        if filtered_job_df.empty:
            st.error(f"Could not find the selected job: '{job_role}' at '{company}'. Please refresh."); return

        selected_job = filtered_job_df.iloc[0]
        jd_skills, job_id = selected_job['skills'], selected_job['id']

        if isinstance(jd_skills, str):
            import json
            try:
                jd_skills = json.loads(jd_skills.replace("'", '"'))
            except json.JSONDecodeError:
                jd_skills = [skill.strip() for skill in jd_skills.strip("[]{}").replace("'", "").replace('"', '').split(',')]
        
        resume_keywords = extract_keywords(text)
        matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)
        phase, status = ("Not Selected", "Rejected") if score < 70 else ("Round 1 (Interview Pending Scheduling)", "In Progress")

        st.markdown("---"); st.subheader("Please Review Your Application")
        st.info(f"Applying for: **{company} — {job_role}**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Calculated Match Score:** {score}%")
            st.write(f"**Parsed Email:** {candidate_email}")
            st.write(f"**Initial Phase:** {phase}")
        with col2:
            st.write(f"**Matched Skills:**"); st.success(f"{', '.join(sorted(matched)) if matched else 'None'}")
            st.write(f"**Missing Skills:**"); st.error(f"{', '.join(sorted(missing)) if missing else 'None'}")
        
        if st.button("Confirm and Submit Application"):
            application_data = {"user_id": st.session_state["user_id"], "job_id": job_id, "match_score": score,
                                "phase": phase, "status": status, "submission_date": datetime.now().isoformat(),
                                "candidate_name": "Candidate", "candidate_email": candidate_email}
            try:
                supabase.table('applications').insert(application_data).execute()
                st.success("Application submitted successfully!"); st.rerun()
            except Exception as e:
                st.error(f"Failed to submit application: {e}")

def hr_view():
    st.header("HR Dashboard")
    
    with st.expander("Add New Job Opening"):
        with st.form("new_job_form", clear_on_submit=True):
            job_title = st.text_input("Job Title")
            company_name = st.text_input("Company Name")
            job_description = st.text_area("Job Description")
            job_skills = st.text_input("Required Skills (comma-separated)")
            
            submitted = st.form_submit_button("Add Job")
            if submitted:
                if job_title and company_name and job_skills:
                    skills_list = [skill.strip().lower() for skill in job_skills.split(',')]
                    new_job_data = {
                        "title": job_title, "company": company_name,
                        "description": job_description, "skills": skills_list
                    }
                    try:
                        supabase.table('jobs').insert(new_job_data).execute()
                        st.success(f"Successfully added job: {job_title} at {company_name}")
                        st.cache_data.clear(); st.rerun()
                    except Exception as e:
                        st.error(f"Failed to add job: {e}")
                else:
                    st.warning("Please fill out all required fields (Title, Company, Skills).")

    df = load_all_applications()
    if df.empty:
        st.warning("No candidates yet."); return

    st.subheader("All Candidates Overview")
    st.dataframe(df.drop(columns=['candidate_name'])) # Remove name column from overview
    
    st.subheader("Process Candidate")
    eligible_df = df[(df['status'] == 'In Progress') & (df['match_score'] >= 70)].copy()
    if eligible_df.empty:
        st.info("No eligible candidates to process."); return

    company_sel = st.selectbox("Filter by Company", sorted(eligible_df["company"].unique()))
    filtered_company = eligible_df[eligible_df["company"] == company_sel]
    role_sel = st.selectbox("Filter by Role", sorted(filtered_company["role"].unique()))
    filtered_role = filtered_company[filtered_company["role"] == role_sel]
    
    if filtered_role.empty:
        st.info(f"No eligible candidates for {role_sel} at {company_sel}."); return

    # Use email as the display identifier
    selected_display = st.selectbox("Select Candidate Email", filtered_role['email'].unique())
    
    if not selected_display: return
        
    candidate = filtered_role[filtered_role['email'] == selected_display].iloc[0]
    app_id = int(candidate['id'])

    st.markdown(f"**Email:** {candidate['email']} | **Score:** {candidate['match_score']}% | **Phase:** {candidate['current_phase']}")
    
    if "Pending Scheduling" in candidate["current_phase"]:
        with st.form(key=f"schedule_form_{app_id}"):
            st.write("Schedule Interview")
            c1, c2, c3 = st.columns(3)
            meeting_date = c1.date_input("Date")
            meeting_time = c2.time_input("Time")
            duration_mins = c3.number_input("Duration (mins)", 15, 240, 30, 15)
            meet_link = st.text_input("Google Meet Link", "https://meet.google.com/...")
            
            if st.form_submit_button("Send Interview Invite"):
                start_dt = datetime.combine(meeting_date, meeting_time)
                # Use generic salutation
                email_body = f"""
Dear Candidate,

This is an invitation for an interview for the {candidate['role']} position at {candidate['company']}.

Date: {start_dt.strftime('%A, %B %d, %Y')}
Time: {start_dt.strftime('%I:%M %p')}
Duration: {duration_mins} minutes
Meeting Link: {meet_link}

Please let us know if you have any questions.

Best regards,
The HR Team
"""
                sent = send_email(candidate["email"], f"Interview Invitation: {candidate['role']} at {candidate['company']}", email_body)

                if sent:
                    new_phase = candidate["current_phase"].replace("Pending Scheduling", "Scheduled")
                    supabase.table('applications').update({'phase': new_phase}).eq('id', app_id).execute()
                    st.success("Interview scheduled!"); st.rerun()

    elif "Scheduled" in candidate["current_phase"]:
        result = st.radio("Result of Current Round", ["Pass", "Fail"], key=f"result_{app_id}", horizontal=True)
        if st.button("Submit Result", key=f"submit_{app_id}"):
            update_data = {}
            if result == "Fail":
                update_data = {"status": "Rejected", "phase": "Rejected"}
            else:
                current_phase = candidate['current_phase']
                if "Round 1" in current_phase: update_data = {"phase": "Round 2 (Interview Pending Scheduling)"}
                elif "Round 2" in current_phase: update_data = {"phase": "Final (Interview Pending Scheduling)"}
                elif "Final" in current_phase:
                    update_data = {"status": "Selected", "phase": "Selected"}
                    # Use generic salutation
                    offer_body = f"Dear Candidate,\n\nCongratulations! You have been selected for the {candidate['role']} position at {candidate['company']}.\n\nWe are excited to welcome you to the team.\n\nBest regards,\nThe HR Team"
                    send_email(candidate["email"], "Job Offer", offer_body)
            
            if update_data:
                supabase.table('applications').update(update_data).eq('id', app_id).execute()
                st.success("Result updated!"); st.rerun()

def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    if "username" not in st.session_state:
        login_register_ui()
    else:
        with st.sidebar:
            st.markdown(f"Logged in as `{st.session_state['username']}`")
            if st.button("Logout", key="logout_btn"):
                st.session_state.clear(); st.rerun()
        
        if st.session_state["role"] == "User": user_view()
        elif st.session_state["role"] == "HR": hr_view()
        else: st.error("Unknown role")

if __name__ == "__main__":
    main()

