import streamlit as st
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import os
# ---------- CONFIG ----------
st.set_page_config(page_title="AI Safety System", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.chat-user {
    background-color: #0084ff;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: right;
}
.chat-bot {
    background-color: #2f2f2f;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("🚧 AI Safety Monitoring + Chatbot")

# ---------- LOAD MODELS ----------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY1"))
model = genai.GenerativeModel('models/gemini-2.0-flash')
model_yolo = YOLO("models/best_ppe_model.pt")
# ---------- CHAT FUNCTION ----------
def rag_chat(question, analysis, image, mode):
    q = question.lower()
    analysis = analysis.lower()

    if "safe" in q:
        if "unsafe" in analysis:
            return "No, the worker is NOT safe because PPE is missing."
        else:
            return "Yes, the worker is safe."

    elif "helmet" in q:
        if "helmet: yes" in analysis:
            return "Helmet is worn."
        else:
            return "Helmet is NOT worn."

    elif "vest" in q:
        if "vest: yes" in analysis:
            return "Safety vest is worn."
        else:
            return "Safety vest is NOT worn."

    return "Based on the analysis, safety guidelines are being followed."
# ---------- FUNCTIONS ----------
def analyze(image):
    prompt = """
    Check PPE compliance.

    Output:
    Helmet: Yes/No
    Vest: Yes/No
    Status: SAFE/UNSAFE
    Reason:
    """

    try:
        response = model.generate_content([prompt, image])
        return response.text, "gemini"

    except Exception:
        return """
Helmet: Unknown
Vest: Unknown
Status: CHECK REQUIRED
Reason: Gemini quota exceeded. Using fallback mode.
""", "fallback"

# ---------- YOLO PPE DETECTION ----------
def detect_ppe(image):
    results = model_yolo(image, conf=0.2)

    detected = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id].lower()

        if label in ["helmet", "vest"]:   # 🔥 filter
            detected.append(label)

    return list(set(detected)) # remove duplicates


def analyze_ppe(detected):
    detected = [d.lower() for d in detected]  # 🔥 FIX

    helmet = "helmet" in detected
    vest = "vest" in detected

    if helmet and vest:
        status = "SAFE"
        reason = "Worker is wearing helmet and safety vest."

    elif not helmet:
        status = "UNSAFE"
        reason = "Helmet not detected."

    elif not vest:
        status = "UNSAFE"
        reason = "Safety vest not detected."

    else:
        status = "CHECK REQUIRED"
        reason = "Unable to determine PPE."

    return f"""
Helmet: {'Yes' if helmet else 'No'}
Vest: {'Yes' if vest else 'No'}
Status: {status}
Reason: {reason}
"""

# ---------- SESSION STATE ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)

    detected = detect_ppe(image)
    analysis = analyze_ppe(detected)

    st.write("Detected:", detected)

    if "SAFE" in analysis:
        st.success(analysis)
    else:
        st.error(analysis)

    results = model_yolo(image)

# Show image with boxes
    annotated = results[0].plot()
    st.image(annotated, caption="Detected Image", use_column_width=True)

    # ---------- CHAT ----------
    st.markdown("---")
    st.subheader("💬 Safety Chatbot")

    user_input = st.text_input("Ask your question")

    if user_input:
        answer = rag_chat(user_input, analysis, image, "yolo")
        # Save chat history
        st.session_state.chat.append(("user", user_input))
        st.session_state.chat.append(("bot", answer))

    # ---------- DISPLAY CHAT ----------
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f'<div class="chat-user">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">{msg}</div>', unsafe_allow_html=True)

else:
    st.info("Upload an image to start")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("🚀 Powered by YOLO + Gemini + RAG")


print(model_yolo.names)