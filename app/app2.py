
import streamlit as st
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import cv2
import os

if "chat_answer" not in st.session_state:
    st.session_state.chat_answer = ""

if "image_explanation" not in st.session_state:
    st.session_state.image_explanation = ""

# ---------- CONFIG ----------
st.set_page_config(page_title="AI PPE Safety System", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #00FFAA;
}
.stButton>button {
    background-color: #00FFAA;
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("🚧 AI-Based PPE Compliance Monitoring System")

# ---------- LOAD MODELS ----------
model_yolo = YOLO("models/best_ppe_model.pt")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY1"))
model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")

# ---------- DETECTION ----------
def detect_ppe(image):
    results = model_yolo(image, conf=0.1)

    detected = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]

        print("Detected:", label)

        if label in ["Helmet", "Vest"]:
            detected.append(label.lower())

    return list(set(detected)), results


# ---------- ANALYSIS ----------
def analyze_ppe(detected):
    helmet = "helmet" in detected
    vest = "vest" in detected

    if helmet and vest:
        return "SAFE", "Worker is fully protected."
    elif not helmet:
        return "UNSAFE", "Helmet not detected."
    elif not vest:
        return "UNSAFE", "Safety vest not detected."
    else:
        return "CHECK", "Unable to determine PPE."

def explain_image(image, status, reason):
    try:
        response = model_gemini.generate_content([
            f"""
            Analyze this construction image.

            Current detection:
            Status: {status}
            Reason: {reason}

            Provide:
            1. Safety evaluation
            2. Missing equipment
            3. Risk level
            4. Recommendation
            """,
            image
        ])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
    

# ---------- RAG ----------
def build_context(status, reason):
    return f"""
    PPE Status: {status}
    Reason: {reason}

    Safety Guidelines:
    - Helmet is mandatory in construction areas.
    - Safety vest improves visibility.
    - PPE prevents serious injuries.
    """
    st.session_state.image_explanation = explain_image(image, status, reason)

def smart_safety_analysis(image, detected, status, reason):
    try:
        # If vest missing → double check with Gemini
        if "vest" not in detected:
            response = model_gemini.generate_content([
                "Is the person wearing any kind of safety vest or harness? Answer YES or NO.",
                image
            ])

            answer = response.text.lower()

            if "yes" in answer:
                return "WARNING", "Possible safety gear detected but not clearly identified."

        return status, reason

    except Exception as e:
        return status, reason


# def rag_chat(question, status, reason):
#     context = build_context(status, reason)

#     prompt = f"""
#     You are a construction safety expert.

#     {context}

#     Answer the question clearly:
#     {question}
#     """

#     try:
#         response = model_gemini.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return str(e)

def rag_chat(question, analysis):
    try:
        prompt = f"""
        You are an expert construction safety AI assistant.

        Context:
        {analysis}

        Instructions:
        - Be clear and professional
        - Give safety advice
        - Mention risks if any

        Question: {question}
        """

        response = model_gemini.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"
    
   

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Webcam"])

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([1, 1])

# ---------- IMAGE MODE ----------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload PPE Image", type=["jpg", "png", "jpeg","webp"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        detected, results = detect_ppe(image)
        status, reason = analyze_ppe(detected)
        status, reason = smart_safety_analysis(image, detected, status, reason)

        annotated = results[0].plot()

        with col1:
            st.image(annotated, caption="Detection Result", use_container_width=True)

        with col2:
            st.subheader("📊 Safety Status")

            if status == "SAFE":
                st.success(f"✅ {status}")
            else:
                st.error(f"❌ {status}")

            st.write(f"**Reason:** {reason}")
            st.write(f"**Detected PPE:** {detected}")

        # ---------- CHAT ----------
        st.markdown("---")
        st.subheader("💬 AI Safety Assistant")

        user_input = st.text_input("Ask about safety...")

        if user_input:
         analysis = f"Status: {status}\nReason: {reason}"
         st.session_state.chat_answer = rag_chat(user_input, analysis)

if st.session_state.chat_answer:
    st.write("🤖", st.session_state.chat_answer)

# ---------- WEBCAM MODE ----------
elif mode == "Webcam":
    st.warning("📷 Click Start to activate webcam")

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()

            if not ret:
                st.error("Camera not working")
                break

            detected, results = detect_ppe(frame)
            status, reason = analyze_ppe(detected)

            annotated = results[0].plot()
            frame_window.image(annotated)

        cap.release()


if mode == "Upload Image" and uploaded_file:
    st.subheader("🧠 AI Image Explanation")

    if st.button("Explain Image"):
        st.session_state.image_explanation = explain_image(image,status,reason)

    if st.session_state.image_explanation:
        st.write(st.session_state.image_explanation)


# if user_input:
#       analysis = f"Status: {status}\nReason: {reason}"
#       answer = rag_chat(user_input, analysis)
#       st.write(answer)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("🚀 Powered by YOLOv8 + Gemini AI | Developed for Smart Safety Monitoring")



















# import streamlit as st
# from ultralytics import YOLO
# import google.generativeai as genai
# from PIL import Image
# import os

# # ---------- CONFIG ----------
# st.set_page_config(page_title="AI PPE Safety System", layout="wide")

# # ---------- TITLE ----------
# st.title("🚧 AI PPE Detection + RAG Chatbot")

# # ---------- LOAD MODELS ----------
# model_yolo = YOLO("best.pt")  # your trained model

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY1"))
# model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")

# # ---------- PPE DETECTION ----------
# def detect_ppe(image):
#     results = model_yolo(image, conf=0.2)

#     detected = []

#     for box in results[0].boxes:
#         cls_id = int(box.cls[0])
#         label = results[0].names[cls_id].lower()

#         if label in ["helmet", "vest"]:
#             detected.append(label)

#     return list(set(detected)), results


# # ---------- SAFETY ANALYSIS ----------
# def analyze_ppe(detected):
#     helmet = "helmet" in detected
#     vest = "vest" in detected

#     if helmet and vest:
#         status = "SAFE"
#         reason = "Worker is wearing helmet and safety vest."

#     elif not helmet:
#         status = "UNSAFE"
#         reason = "Helmet not detected."

#     elif not vest:
#         status = "UNSAFE"
#         reason = "Safety vest not detected."

#     else:
#         status = "CHECK REQUIRED"
#         reason = "Unable to determine PPE."

#     return f"""
# Helmet: {'Yes' if helmet else 'No'}
# Vest: {'Yes' if vest else 'No'}
# Status: {status}
# Reason: {reason}
# """


# # ---------- RAG CONTEXT ----------
# def build_context(analysis):
#     return f"""
#     Safety Analysis Report:

#     {analysis}

#     Safety Rules:
#     - Helmet is mandatory in construction zones.
#     - Safety vest improves visibility.
#     - Missing PPE can lead to serious injury.
#     """


# # ---------- GEMINI CHAT ----------
# def rag_chat(question, analysis):
#     try:
#         prompt = f"{analysis}\n\nQuestion: {question}"

#         response = model_gemini.generate_content(prompt)
#         return response.text

#     except Exception as e:
#         return f"⚠️ Gemini error: {str(e)}"


# # ---------- UI ----------
# uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg","webp"])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")

#     # Detection
#     detected, results = detect_ppe(image)
#     analysis = analyze_ppe(detected)

#     # Show image
#     annotated = results[0].plot()
#     st.image(annotated, caption="Detected Image", use_column_width=True)

#     # Show result
#     st.subheader("🧠 Detection Result")
#     st.write("Detected:", detected)

#     if "SAFE" in analysis:
#         st.success(analysis)
#     else:
#         st.error(analysis)

#     # ---------- CHATBOT ----------
#     st.markdown("---")
#     st.subheader("💬 Safety Chatbot (RAG + Gemini)")

#     user_input = st.text_input("Ask a question")

#     if user_input:
#         answer = rag_chat(user_input, analysis)
#         st.write("🤖", answer)

# else:
#     st.info("Upload an image to start")

# # ---------- FOOTER ----------
# st.markdown("---")
# st.markdown("🚀 YOLO + RAG + Gemini System")