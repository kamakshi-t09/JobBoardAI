import streamlit as st
import sqlite3
import os
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document


# -------------------------------------------------
# NLTK SETUP
# -------------------------------------------------
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# CONFIG

# -------------------------------------------------
st.set_page_config(
    page_title="JobBoardAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# PROFESSIONAL UI CSS
# -------------------------------------------------
st.markdown("""
<style>
header, footer, #MainMenu {display: none !important;}
section[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}

html, body {
    font-family: 'Inter','Segoe UI',sans-serif;
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 0.5rem !important;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#020617);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

.app-header {
    background: linear-gradient(135deg,#1e3a8a,#2563eb);
    padding: 14px 22px;
    border-radius: 14px;
    color: white;
    margin-bottom: 18px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 10px 26px rgba(0,0,0,0.06);
    margin-bottom: 22px;
}

input, textarea {
    border-radius: 10px !important;
    padding: 10px !important;
}

.stButton > button {
    background: linear-gradient(135deg,#2563eb,#1e40af);
    color: white;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PATHS
# -------------------------------------------------
DB_NAME = "jobboard.db"
UPLOAD_RESUME = "uploads/resumes"
UPLOAD_PROFILE = "uploads/profile_resumes"
os.makedirs(UPLOAD_RESUME, exist_ok=True)
os.makedirs(UPLOAD_PROFILE, exist_ok=True)

# -------------------------------------------------
# DATABASE
# -------------------------------------------------
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT,
    role TEXT,
    phone TEXT,
    skills TEXT,
    qualification TEXT,
    dob TEXT,
    address TEXT,
    resume TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS jobs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_email TEXT,
    title TEXT,
    company TEXT,
    location TEXT,
    salary TEXT,
    vacancies INTEGER,
    description TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS applications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    applicant_email TEXT,
    name TEXT,
    phone TEXT,
    resume TEXT
)
""")

conn.commit()

def add_column(table, column, dtype):
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
        conn.commit()
    except sqlite3.OperationalError:
        pass

for col in ["phone","skills","qualification","dob","address","resume"]:
    add_column("users", col, "TEXT")


# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])

def ats_score(resume_text, job_text):
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    semantic = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    resume_words = set(resume_text.split())
    job_words = set(job_text.split())
    keyword_match = len(resume_words & job_words) / len(job_words) if job_words else 0

    return round((semantic * 0.7 + keyword_match * 0.3) * 100, 2)

def extract_keywords(text, top_n=15):
    words = clean_text(text).split()
    freq = {}
    for w in words:
        if len(w) > 2:
            freq[w] = freq.get(w,0)+1
    return [k for k,_ in sorted(freq.items(), key=lambda x:x[1], reverse=True)[:top_n]]

def recommend_jobs(resume_text, jobs_df):
    resume_keywords = set(extract_keywords(resume_text))
    results = []
    for _, job in jobs_df.iterrows():
        overlap = len(resume_keywords & set(clean_text(job["description"]).split()))
        ats = ats_score(resume_text, job["description"])
        results.append({**job, "ats": ats, "overlap": overlap})
    return sorted(results, key=lambda x:x["ats"], reverse=True)[:5]

def search_jobs(query):
    like = f"%{query}%"
    return pd.read_sql("""
    SELECT * FROM jobs
    WHERE title LIKE ? OR company LIKE ? OR location LIKE ? OR description LIKE ?
    """, conn, params=(like,like,like,like))


# -------------------------------------------------
# SESSION INITIALIZATION (CRITICAL FIX)
# -------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if "apply_job" not in st.session_state:
    st.session_state.apply_job = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"

if "profile_edit" not in st.session_state:
    st.session_state.profile_edit = True

# ✅ ADD THIS
if "choice" not in st.session_state:
    st.session_state.choice = "Home"

# -------------------- SIDEBAR --------------------
st.sidebar.title("JobBoardAI")

if st.session_state.user:
    st.session_state.choice = st.sidebar.radio(
        "Navigation",
        ["Home", "Jobs", "Resume Match", "Applications", "Profile"],
        index=["Home", "Jobs", "Resume Match", "Applications", "Profile"]
              .index(st.session_state.choice)
    )


# -------------------- AUTH (MAIN PAGE) --------------------
if st.session_state.user is None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("Welcome to JobBoardAI")

    auth = st.radio(
        "Account",
        ["Login", "Signup"],
        horizontal=True,
        key="auth_mode"
    )

    if auth == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            cur.execute(
                "SELECT * FROM users WHERE email=? AND password=?",
                (email, password)
            )
            user = cur.fetchone()
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["job_seeker", "recruiter"])

        if st.button("Create Account"):
            try:
                cur.execute(
                    "INSERT INTO users(name,email,password,role) VALUES (?,?,?,?)",
                    (name, email, password, role)
                )
                conn.commit()
                st.success("Account created. Please login.")
                
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("Email already exists")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# PROFILE PAGE (FIXED & PROFESSIONAL)
# -------------------------------------------------
else:
    st.markdown("""
    <div class="app-header">
        <h2 style="margin:0;">JobBoardAI</h2>
        <span style="font-size:13px;">AI-powered job search & resume matching</span>
    </div>
    """, unsafe_allow_html=True)

    

     # ---------------- HOME ----------------
    if st.session_state.choice == "Home":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        search = st.text_input("🔍 Search jobs posted by recruiters")
        if search:
            jobs = search_jobs(search)
            for _, job in jobs.iterrows():
                st.subheader(f"{job['title']} – {job['company']}")
                st.write(job["description"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- JOBS ----------------
    elif st.session_state.choice == "Jobs":
        role = st.session_state.user[4]
        email = st.session_state.user[2]

        # ---------------- RECRUITER VIEW ----------------
        if role == "recruiter":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📢 Post a Job")

            title = st.text_input("Job Title")
            company = st.text_input("Company Name")
            location = st.text_input("Location")
            salary = st.text_input("Salary")
            vacancies = st.number_input("Vacancies", min_value=1)
            desc = st.text_area("Job Description")

            if st.button("Post Job"):
                cur.execute("""
                INSERT INTO jobs(recruiter_email,title,company,location,salary,vacancies,description)
                VALUES (?,?,?,?,?,?,?)
                """, (email, title, company, location, salary, vacancies, desc))
                conn.commit()
                st.success("Job posted successfully")
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

            # -------- Recruiter's Posted Jobs --------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📄 Your Posted Jobs")

            my_jobs = pd.read_sql(
                "SELECT * FROM jobs WHERE recruiter_email=?",
                conn, params=(email,)
            )

            if my_jobs.empty:
                st.info("You haven't posted any jobs yet.")
            else:
                for _, job in my_jobs.iterrows():
                    st.markdown(f"### {job['title']} – {job['company']}")
                    st.write(job["description"])
                    if st.button("Delete Job", key=f"del_{job['id']}"):
                        cur.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
                        conn.commit()
                        st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

            # -------- All Jobs (Read-only) --------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🌍 All Available Jobs")

            all_jobs = pd.read_sql("SELECT * FROM jobs", conn)

            for _, job in all_jobs.iterrows():
                st.markdown(f"**{job['title']} – {job['company']}**")
                st.caption(job["location"])
                st.write(job["description"])
                st.markdown("---")

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- JOB SEEKER VIEW ----------------
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💼 Available Jobs")

            jobs = pd.read_sql("SELECT * FROM jobs", conn)

            for _, job in jobs.iterrows():
                st.markdown(f"### {job['title']} – {job['company']}")
                st.write(job["description"])

                if st.button("Apply", key=f"apply_{job['id']}"):
                    st.session_state.apply_job = job["id"]
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    
     # ---------------- APPLY JOB ----------------
    if st.session_state.apply_job:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Job Application")

        cur.execute("SELECT title, company FROM jobs WHERE id=?",
                    (st.session_state.apply_job,))
        job = cur.fetchone()
        st.write(f"Applying for **{job[0]} – {job[1]}**")

        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])

        if st.button("Submit Application"):
            if not resume:
                st.error("Please upload resume")
            else:
                resume_path = os.path.join(UPLOAD_RESUME, resume.name)
                with open(resume_path, "wb") as f:
                    f.write(resume.read())

                cur.execute("""
                INSERT INTO applications(job_id, applicant_email, name, phone, resume)
                VALUES (?,?,?,?,?)
                """, (
                    st.session_state.apply_job,
                    st.session_state.user[2],
                    name,
                    phone,
                    resume_path
                ))
                conn.commit()

                st.success("Application submitted successfully 🎉")
                st.session_state.apply_job = None
                st.rerun()

        if st.button("Cancel"):
            st.session_state.apply_job = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


    # ---------------- RESUME MATCH ----------------
    elif st.session_state.choice == "Resume Match":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        resume = st.file_uploader("Upload Resume", type=["pdf","docx"])
        if resume:
            text = extract_text(resume)
            st.write("**Keywords:**", ", ".join(extract_keywords(text)))
            jobs = pd.read_sql("SELECT * FROM jobs", conn)
            for job in recommend_jobs(text, jobs):
                st.subheader(f"{job['title']} – {job['company']}")
                st.write(f"ATS Score: {job['ats']}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- APPLICATIONS ----------------
    elif st.session_state.choice == "Applications":
        email = st.session_state.user[2]
        role = st.session_state.user[4]

        st.markdown("<div class='section-title'>📄 Applications</div>", unsafe_allow_html=True)

        if role == "job_seeker":
            df = pd.read_sql("""
                SELECT 
                    a.id,
                    j.title AS job_title,
                    j.company AS company_name,
                    a.name,
                    a.phone,
                    a.resume
                FROM applications a
                JOIN jobs j ON a.job_id = j.id
                WHERE a.applicant_email = ?
            """, conn, params=(email,))
        else:
            df = pd.read_sql("""
                SELECT 
                    a.id,
                    j.title AS job_title,
                    j.company AS company_name,
                    a.applicant_email,
                    a.name,
                    a.phone,
                    a.resume
                FROM applications a
                JOIN jobs j ON a.job_id = j.id
                WHERE j.recruiter_email = ?
            """, conn, params=(email,))

        if df.empty:
            st.info("No applications found.")
        else:
            for _, row in df.iterrows():
                st.markdown("<div class='app-card'>", unsafe_allow_html=True)

                # 🔹 Job + Company
                st.markdown(
                    f"<div class='app-title'>{row['job_title']} – {row['company_name']}</div>",
                    unsafe_allow_html=True
                )

                # 🔹 Candidate info
                st.markdown(
                    f"<div class='app-meta'>👤 {row['name']} | 📞 {row['phone']}</div>",
                    unsafe_allow_html=True
                )

                # 🔹 Resume
                resume_path = row["resume"]
                if resume_path and os.path.exists(resume_path):
                    with open(resume_path, "rb") as f:
                        st.download_button(
                            "📥 Download Resume",
                            data=f,
                            file_name=os.path.basename(resume_path),
                            mime="application/pdf",
                            key=f"resume_{row['id']}"
                        )
                else:
                    st.warning("Resume not available")

                st.markdown("</div>", unsafe_allow_html=True)






    elif st.session_state.choice == "Profile":
        st.markdown('<div class="card">', unsafe_allow_html=True)

        cur.execute("SELECT * FROM users WHERE email=?", (st.session_state.user[2],))
        user = cur.fetchone()

        disabled = not st.session_state.profile_edit

        st.subheader("👤 Profile")

        name = st.text_input("Name", user[1], disabled=disabled)
        st.text_input("Email", user[2], disabled=True)
        phone = st.text_input("Phone", user[5] or "", disabled=disabled)
        qualification = st.text_input("Qualification", user[7] or "", disabled=disabled)
        dob = st.text_input("DOB", user[8] or "", disabled=disabled)
        address = st.text_area("Address", user[9] or "", disabled=disabled)

        resume_path = user[10]
        if st.session_state.profile_edit:
            resume = st.file_uploader("Upload Resume", type=["pdf","docx"])
            if resume:
                resume_path = os.path.join(UPLOAD_PROFILE, resume.name)
                with open(resume_path, "wb") as f:
                    f.write(resume.read())

        if st.session_state.profile_edit:
            if st.button("💾 Save Profile"):
                cur.execute("""
                UPDATE users
                SET name=?, phone=?, qualification=?, dob=?, address=?, resume=?
                WHERE email=?
                """, (name, phone, qualification, dob, address, resume_path, user[2]))
                conn.commit()
                st.session_state.profile_edit = False
                st.success("Profile saved successfully")
                st.rerun()
        else:
            if st.button("✏️ Edit Profile"):
                st.session_state.profile_edit = True
                st.rerun()

        if st.button("Logout"):
              st.session_state.user = None
              st.session_state.profile_edit = True
              if "auth_mode" in st.session_state:
                del st.session_state["auth_mode"]

              st.rerun()



        st.markdown('</div>', unsafe_allow_html=True)
