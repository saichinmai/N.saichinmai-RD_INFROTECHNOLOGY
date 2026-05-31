import streamlit as st

st.title("AI-Powered Medical Diagnosis System")

st.write("Select symptoms to predict disease")

# Symptoms
fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
vomiting = st.checkbox("Vomiting")
fatigue = st.checkbox("Fatigue")
body_pain = st.checkbox("Body Pain")
cold = st.checkbox("Cold")
chest_pain = st.checkbox("Chest Pain")
breathing_problem = st.checkbox("Breathing Problem")

if st.button("Predict Disease"):

    disease = "Unable to Predict"

    # Better prediction logic
    if fever and cough and headache and body_pain:
        disease = "Flu"

    elif fever and cough and cold:
        disease = "Cold"

    elif headache and not fever:
        disease = "Migraine"

    elif vomiting and fever and headache:
        disease = "Food Poisoning"

    elif fatigue and not fever and not cough:
        disease = "Diabetes"

    elif chest_pain and fatigue:
        disease = "Heart Disease"

    elif cough and breathing_problem:
        disease = "Respiratory Infection"

    elif fever and body_pain:
        disease = "Viral Fever"

    st.success(f"Predicted Disease: {disease}")