import streamlit as st
from PIL import Image
import numpy as np

# Importing CLIP and LLaMA utilities
from src.clip_utils import load_clip_model, get_text_features, get_image_features, compute_similarity, load_custom_clip_model
from src.llama_utils import process_user_input, generate_clip_description
from src.classes import get_candidate_captions
from src.oauth import get_login_url, get_user_info
from src.firebase_config import create_user_if_not_exists, create_new_chat, get_user_chats, load_chat_history  

def main():
    st.set_page_config(page_title="CLIP Crop & Disease Detection", layout="wide")

    # Load models
    model, preprocess, device = load_clip_model()
    candidate_captions = get_candidate_captions()

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ==== SIDEBAR (Chat History + Google Sign-In) ====
    with st.sidebar:
        st.title("Account")

        # Google Sign-In Section
        if "google_user" not in st.session_state:
            login_url = get_login_url()
            st.markdown(f'<a href="{login_url}" target="_self"><button style="width: 100%;">🔑 Sign in with Google</button></a>', unsafe_allow_html=True)

            query_params = st.query_params
            # Check if user has attempted to log in ('code' in url)
            if "code" in query_params:
                auth_code = query_params["code"]
                user_info = get_user_info(auth_code)
                if user_info:
                    st.session_state["google_user"] = user_info
    
                    # Create user entry in Firestore if not exists
                    user_id = user_info["sub"]
                    create_user_if_not_exists(user_id, user_info["name"], user_info["email"])
    
                    # Add a new chat session
                    new_chat_id = create_new_chat(user_id)
                    st.session_state["current_chat_id"] = new_chat_id
    
                    st.rerun()
        else:
            user = st.session_state["google_user"]
            col1, col2 = st.columns([1, 4])  # Align profile picture and text
            with col1:
                st.image(user["picture"], width=40)
            with col2:
                st.write(f"**{user['name']}**")
                st.write(f"📧 {user['email']}")
            
            if st.button("Logout", use_container_width=True):
                st.session_state.pop("google_user", None)
                st.rerun()

        st.divider()  # Separator for better layout

        # Display Chat History in Sidebar
        st.subheader("📝 Chat History")
        for message in st.session_state.chat_history:
            role_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
            st.write(f"{role_icon} {message['content']}")

    # ==== MAIN CONTENT ====
    st.title("🌱 CLIP Crop & Disease Detection")
    st.write("Upload an image and let AI detect potential crop diseases and provide insights.")

    # Image Upload & Processing
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Get features and compute similarities
        text_features = get_text_features(candidate_captions, device, model)
        image_features = get_image_features(image, device, model, preprocess)
        similarities = compute_similarity(image_features, text_features)

        # Get the top 3 most similar captions
        top_indices = np.argsort(similarities.cpu().numpy())[::-1][:3] 
        top_captions = [candidate_captions[idx] for idx in top_indices]
        top_confidences = [similarities[idx].item() * 100 for idx in top_indices]

        best_caption = top_captions[0]
        confidence = top_confidences[0]

        # Generate AI response if no history exists
        if not st.session_state.chat_history:
            generate_clip_description(st, best_caption, confidence)

    # Chat Interaction
    st.subheader("💬 Chat with LLAMA")
    user_prompt = st.chat_input("Ask LLAMA about this diagnosis...")
    if user_prompt:
        process_user_input(st, user_prompt)  # Handle user query

if __name__ == "__main__":
    main()