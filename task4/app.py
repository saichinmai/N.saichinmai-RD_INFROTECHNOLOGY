from flask import Flask, render_template, request
import fitz
import os

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = "uploads"

# Create uploads folder automatically
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Job skill database
job_skills = {

    "AI/ML Intern": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "numpy",
        "pandas",
        "nlp",
        "opencv"
    ],

    "Web Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "django",
        "flask"
    ],

    "Data Analyst": [
        "excel",
        "sql",
        "python",
        "power bi",
        "tableau",
        "statistics"
    ]
}


# Extract text from PDF
def extract_text_from_pdf(pdf_path):

    text = ""

    try:
        document = fitz.open(pdf_path)

        for page in document:
            text += page.get_text()

        return text.lower()

    except Exception:
        return ""


# Resume screening logic
def screen_resume(text, role):

    required_skills = job_skills.get(role, [])

    matched_skills = []

    for skill in required_skills:
        if skill.lower() in text:
            matched_skills.append(skill)

    # Calculate score
    if len(required_skills) > 0:
        score = int(
            (len(matched_skills) /
             len(required_skills)) * 100
        )
    else:
        score = 0

    # Selection logic
    if score >= 60:
        result = "Selected"
    else:
        result = "Rejected"

    return score, matched_skills, result


@app.route("/", methods=["GET", "POST"])
def home():

    score = None
    skills = []
    result = ""

    if request.method == "POST":

        role = request.form["role"]

        uploaded_file = request.files["resume"]

        if uploaded_file.filename != "":

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                uploaded_file.filename
            )

            uploaded_file.save(filepath)

            resume_text = extract_text_from_pdf(
                filepath
            )

            score, skills, result = screen_resume(
                resume_text,
                role
            )

    return render_template(
        "index.html",
        score=score,
        skills=skills,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)