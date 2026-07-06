
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

# Stores chatbot response
# Stores image explanation
# Prevents Streamlit refresh issue
# Keeps outputs persistent




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
}st.session_state.image_explanation = explain_image(image, status, reason)
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("🚧 AI-Based PPE Compliance Monitoring System")

# ---------- LOAD MODELS ----------
model_yolo = YOLO("models/best_ppe_model.pt")

genai.configure(api_key=os.getenv("Gemini_api_key"))
model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")

# Loads trained YOLO model
# Connects Gemini API
# Initializes LLM for AI reasoning
# ---------- DETECTION ----------
def detect_ppe(image):
    results = model_yolo(image, conf=0.6)

# Runs object detection on image
# Confidence threshold = 0.6
# Detects helmet and vest

    detected = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]


#Loops through detected objects
# Extracts class label (Helmet / Vest)


        print("Detected:", label)

        if label in ["Helmet", "Vest"]:
            detected.append(label.lower())

    return list(set(detected)), results

# Loops through detected objects
# Extracts class label (Helmet / Vest

# ---------- ANALYSIS ----------
# Removes duplicate detections
# Returns PPE + bounding boxes


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

#

def explain_image(image, status, reason):
    try:
        response = model_gemini.generate_content([
            f"""
You are a construction safety inspector AI.

Detected Result:
Status: {status}
Reason: {reason}

Instructions:
- Keep answer SHORT
- Max 5 bullet points
- Focus only on visible PPE
- Mention missing PPE if any

Format:
- PPE detected
- Missing PPE
- Risk level
- Final comment
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



def rag_chat(question, analysis):
    try:
        prompt = f"""
You are an AI safety assistant for construction.

Context:
{analysis}

Instructions:
- Answer in MAX 3-4 lines
- Be specific to this image
- Do NOT give long theory
- Give direct answer + 1 suggestion

Format:
Answer:
Suggestion:

Question: {question}
"""

        response = model_gemini.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"
   

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Webcam"])

# Takes user query
# Uses detection result as context

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([1, 1])

# ---------- IMAGE MODE ----------
# Load image
# Detect PPE
# Analyze safety

if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload PPE Image", type=["jpg", "png", "jpeg","webp"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        detected, results = detect_ppe(image)
        status, reason = analyze_ppe(detected)

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



# Combines detection result
# Sends to Gemini
# Stores response

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
        st.session_state.image_explanation = explain_image(image, status, reason)

    if st.session_state.image_explanation:
        st.write(st.session_state.image_explanation)


st.markdown("---")
st.markdown("🚀 Powered by YOLOv8 + Gemini AI | Developed for Smart Safety Monitoring")


# ---------------------------------------------------------------------------------------