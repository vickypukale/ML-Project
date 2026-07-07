"""
app.py
------
Streamlit UI for the Career Path Prediction & Guidance System.

Run with:
    streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import (
    load_artifacts, encode_features, NUMERIC_COLS, BINARY_MAPS,
    ORDINAL_MAPS, NOMINAL_COLS, TARGET_COL
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Career Path Prediction & Guidance System",
                    page_icon="🎓", layout="centered")

# ----------------------------------------------------------------------
# Recommended courses per predicted career (rule-based, IF/lookup style)
# ----------------------------------------------------------------------
COURSE_RECOMMENDATIONS = {
    "Applications Developer": [
        "Java Programming Masterclass",
        "Object-Oriented Design & Design Patterns",
        "Android / iOS App Development Fundamentals",
    ],
    "CRM Technical Developer": [
        "Salesforce Platform Developer I",
        "Apex Programming for Salesforce",
        "CRM Systems & Customer Data Modeling",
    ],
    "Database Developer": [
        "SQL for Data Engineering",
        "Database Design & Normalization",
        "PL/SQL / T-SQL Advanced Programming",
    ],
    "Mobile Applications Developer": [
        "Flutter & Dart Complete Guide",
        "iOS Development with Swift",
        "Android Development with Kotlin",
    ],
    "Network Security Engineer": [
        "CompTIA Security+ Certification Prep",
        "Network Defense & Ethical Hacking",
        "CCNA - Cisco Networking Fundamentals",
    ],
    "Software Developer": [
        "Data Structures & Algorithms",
        "Full Stack Web Development",
        "Git, CI/CD & Clean Code Practices",
    ],
    "Software Engineer": [
        "Software Engineering & System Design",
        "Design Patterns in Practice",
        "Agile & Scrum Fundamentals",
    ],
    "Software Quality Assurance (QA) / Testing": [
        "ISTQB Foundation Level Certification",
        "Selenium & Test Automation",
        "API Testing with Postman",
    ],
    "Systems Security Administrator": [
        "CompTIA Security+ Certification Prep",
        "Linux System Administration & Hardening",
        "Identity & Access Management Fundamentals",
    ],
    "Technical Support": [
        "IT Support Professional Certificate",
        "Troubleshooting & Customer Communication",
        "ITIL Foundation Certification",
    ],
    "UX Designer": [
        "UI/UX Design Specialization",
        "Figma for Product Design",
        "Design Thinking & User Research",
    ],
    "Web Developer": [
        "The Complete Web Developer Bootcamp",
        "React.js & Modern JavaScript",
        "Responsive Web Design (HTML/CSS)",
    ],
}


@st.cache_resource
def load_model_and_artifacts():
    encoders, scaler, feature_cols = load_artifacts()
    with open(os.path.join(MODEL_DIR, "best_model_meta.json")) as f:
        meta = json.load(f)

    if meta["is_keras"]:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "best_model.keras"))
    else:
        model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

    return model, encoders, scaler, feature_cols, meta


def predict_career(model, encoders, scaler, feature_cols, meta, raw_input_df):
    X, _, _, _ = encode_features(raw_input_df, encoders, fit_scaler=False, scaler=scaler)
    X = X[feature_cols]

    if meta["is_keras"]:
        probs = model.predict(X.values, verbose=0)[0]
    else:
        probs = model.predict_proba(X)[0]

    target_classes = encoders[TARGET_COL].classes_
    top_idx = np.argsort(probs)[::-1][:3]
    top_predictions = [(target_classes[i], float(probs[i])) for i in top_idx]
    return top_predictions


def main():
    st.title("🎓 Career Path Prediction & Guidance System")
    st.write(
        "Fill in the fields below based on your academic profile, skills "
        "and interests. The trained model will suggest the career paths "
        "that best match your profile, along with recommended courses."
    )

    model, encoders, scaler, feature_cols, meta = load_model_and_artifacts()

    st.caption(f"Model in use: **{meta['best_model_name']}**  "
               f"(test accuracy: {meta['metrics']['accuracy']*100:.1f}%)")

    with st.form("student_form"):
        st.subheader("1. Ratings (1-10 / 0-10 scale, from dataset)")
        c1, c2 = st.columns(2)
        logical_quotient = c1.slider("Logical quotient rating", 0, 10, 5)
        hackathons = c2.slider("Number of hackathons attended", 0, 10, 2)
        coding_skills = c1.slider("Coding skills rating", 0, 10, 5)
        public_speaking = c2.slider("Public speaking points", 0, 10, 5)

        st.subheader("2. Skills & habits")
        c1, c2 = st.columns(2)
        self_learning = c1.selectbox("Self-learning capability?", ["yes", "no"])
        extra_courses = c2.selectbox("Extra courses did?", ["yes", "no"])
        reading_writing = c1.selectbox("Reading and writing skills",
                                        ["poor", "medium", "excellent"])
        memory_score = c2.selectbox("Memory capability score",
                                     ["poor", "medium", "excellent"])
        seniors_input = c1.selectbox("Taken inputs from seniors or elders?",
                                      ["yes", "no"])
        team_work = c2.selectbox("Worked in teams ever?", ["yes", "no"])
        introvert = c1.selectbox("Introvert?", ["yes", "no"])
        worker_type = c2.selectbox("Hard worker or smart worker?",
                                    ["hard worker", "smart worker"])
        mgmt_tech = c1.selectbox("Management or Technical?",
                                  ["Management", "Technical"])

        st.subheader("3. Interests & background")
        certifications = st.selectbox("Certifications completed", [
            "information security", "shell programming", "r programming",
            "distro making", "machine learning", "full stack", "hadoop",
            "app development", "python",
        ])
        workshops = st.selectbox("Workshops attended", [
            "testing", "database security", "game development",
            "data science", "system designing", "hacking",
            "cloud computing", "web technologies",
        ])
        interested_subjects = st.selectbox("Interested subjects", [
            "programming", "Management", "data engineering", "networks",
            "Software Engineering", "cloud computing", "parallel computing",
            "IOT", "Computer Architecture", "hacking",
        ])
        career_area = st.selectbox("Interested career area", [
            "testing", "system developer", "Business process analyst",
            "security", "developer", "cloud computing",
        ])
        company_type = st.selectbox("Type of company want to settle in?", [
            "BPA", "Cloud Services", "product development",
            "Testing and Maintainance Services", "SAaS services",
            "Web Services", "Finance", "Sales and Marketing",
            "Product based", "Service Based",
        ])
        book_type = st.selectbox("Interested type of books", [
            "Action and Adventure", "Anthology", "Art", "Autobiographies",
            "Biographies", "Childrens", "Comics", "Cookbooks", "Diaries",
            "Dictionaries", "Drama", "Encyclopedias", "Fantasy", "Guide",
            "Health", "History", "Horror", "Journals", "Math", "Mystery",
            "Poetry", "Prayer books", "Religion-Spirituality", "Romance",
            "Satire", "Science", "Science fiction", "Self help", "Series",
            "Travel", "Trilogy",
        ])

        submitted = st.form_submit_button("Predict my career path")

    if submitted:
        raw_input = pd.DataFrame([{
            "Logical quotient rating": logical_quotient,
            "hackathons": hackathons,
            "coding skills rating": coding_skills,
            "public speaking points": public_speaking,
            "self-learning capability?": self_learning,
            "Extra-courses did": extra_courses,
            "certifications": certifications,
            "workshops": workshops,
            "reading and writing skills": reading_writing,
            "memory capability score": memory_score,
            "Interested subjects": interested_subjects,
            "interested career area": career_area,
            "Type of company want to settle in?": company_type,
            "Taken inputs from seniors or elders": seniors_input,
            "Interested Type of Books": book_type,
            "Management or Technical": mgmt_tech,
            "hard/smart worker": worker_type,
            "worked in teams ever?": team_work,
            "Introvert": introvert,
        }])

        top_predictions = predict_career(
            model, encoders, scaler, feature_cols, meta, raw_input)

        st.success("Prediction complete!")
        st.subheader("🏆 Recommended career path")
        best_career, best_prob = top_predictions[0]
        st.markdown(f"### {best_career}  \n*Confidence: {best_prob*100:.1f}%*")

        st.subheader("Other close matches")
        for career, prob in top_predictions[1:]:
            st.write(f"- {career} ({prob*100:.1f}%)")

        st.subheader("📚 Recommended courses for you")
        courses = COURSE_RECOMMENDATIONS.get(best_career, [])
        if courses:
            for course in courses:
                st.write(f"- {course}")
        else:
            st.write("No specific course mapping found for this role yet.")

        st.info(
            "Note: this synthetic training dataset shows very weak "
            "statistical relationships between the input features and the "
            "target career label (near-random accuracy on held-out data). "
            "Treat the prediction as illustrative of the end-to-end system "
            "rather than a reliable real-world recommendation."
        )


if __name__ == "__main__":
    main()
