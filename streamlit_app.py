import streamlit as st
from PIL import Image
import numpy as np

# Importing CLIP and LLaMA utilities
from src.clip_utils import load_clip_model, get_text_features, get_image_features, compute_similarity, load_custom_clip_model
from src.llama_utils import process_user_input, generate_clip_description
from src.classes import get_candidate_captions
from src.oauth import get_login_url, get_user_info

def main():
    st.set_page_config(page_title="CLIP Crop & Disease Detection", layout="wide")
    clip_file_path = "clip_finetuned(orange_long).pth"

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    model, preprocess, device = load_clip_model()
    candidate_captions = get_candidate_captions()

    # Layout: Two columns (Main app + Sidebar for Google Sign-In)
    col1, col2 = st.columns([3, 1])  # Adjust ratio for layout

    # ==== Google Sign-In in Sidebar (Right Side) ====
    with col2:
        st.subheader("🔑 Google Sign-In")

        if "google_user" not in st.session_state:
            login_url = get_login_url()
            st.markdown(f'<a href="{login_url}" target="_self"><button>Sign in with Google</button></a>', unsafe_allow_html=True)

            # ✅ Updated: Use `st.query_params` instead of `st.experimental_get_query_params`
            query_params = st.query_params
            if "code" in query_params:
                auth_code = query_params["code"]
                user_info = get_user_info(auth_code)
                st.session_state["google_user"] = user_info
                st.rerun()
        else:
            user = st.session_state["google_user"]
            st.image(user["picture"], width=50)
            st.write(f"**Hello, {user['name']}!**")
            st.write(f"📧 {user['email']}")

            if st.button("Logout"):
                st.session_state.pop("google_user", None)
                st.rerun()

    # ==== Main Content ====
    with col1:
        st.title("🌱 CLIP Crop & Disease Detection")

        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)  # Upload and show image
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Get features for captions and image then compute similarities
            text_features = get_text_features(candidate_captions, device, model)
            image_features = get_image_features(image, device, model, preprocess)
            similarities = compute_similarity(image_features, text_features)
            
            # Get the top 3 most similar captions
            top_indices = np.argsort(similarities.cpu().numpy())[::-1][:3] 
            top_captions = [candidate_captions[idx] for idx in top_indices]
            top_confidences = [similarities[idx].item() * 100 for idx in top_indices]  # Convert to percentage
            
            best_caption = top_captions[0]
            best_class = best_caption.split(":")[0]
            confidence = top_confidences[0]

            # Prompt LLM for description of image
            if len(st.session_state.chat_history) == 0:
                generate_clip_description(st, best_caption, confidence)

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            elif message["role"] == "assistant":
                st.chat_message("assistant").write(message["content"])

        # Input field for user's message:
        user_prompt = st.chat_input("Ask LLAMA...")
    
        if user_prompt:
            process_user_input(st, user_prompt)  # RAG enabled

if __name__ == "__main__":
    main()
