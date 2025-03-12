import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
# Importing CLIP and LLaMA utilities
from src.clip_utils import load_custom_clip_model, get_text_features, get_image_features, classify_image, compute_similarity
from src.classes import get_classes 
from src.oauth import get_login_url, get_google_info
from src.firebase_config import create_user_if_not_exists, create_new_chat, fetch_chat_history, load_chat, update_chat_history, add_feedback
from src.llama_utils import generate_clip_description, process_user_input, display_current_chat, generate_chat_title

def main():
    st.set_page_config(page_title="CLIP Crop & Disease Detection", layout="wide")
    #classes = get_classes()
    clicked_previous_chat = False  # temporary
    prompts = 0  # Track # of prompts

    # Initialize session state
    if "current_chat_history" not in st.session_state:
        st.session_state["current_chat_history"] = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # ==== SIDEBAR (Chat History + Google Sign-In) ====
    with st.sidebar:
        st.title("Account")
        user = None

        language = st.selectbox("Select Language", ["English", "Español", "French"])
        
        if "google_user" not in st.session_state:
            login_url = get_login_url()
            st.markdown(f'<a href="{login_url}" target="_self"><button style="width: 100%;">🔑 Sign in with Google</button></a>', unsafe_allow_html=True)

            query_params = st.query_params
            if "code" in query_params:
                auth_code = query_params["code"]
                user_info = get_google_info(auth_code)
                if user_info:
                    user_id = user_info["sub"]
                    create_user_if_not_exists(user_id, user_info["name"], user_info["email"])
                    user_info["chat_history"] = fetch_chat_history(user_id)
                    st.session_state["google_user"] = user_info
                    st.rerun()
        else:
            st.markdown(
                 f"""
                 <style>
                 div[data-testid="stButton"] > button {{
                     width: 100%;
                     display: block;
                     overflow: hidden;
                     white-space: nowrap;
                     /* padding: 12px 16px; */
                     text-overflow: ellipsis;
                 }}
                 </style>
                 """,
                 unsafe_allow_html=True
            )
            user = st.session_state["google_user"]
            col1, col2 = st.columns([1, 4])
            with col1:
                profile_picture = user.get("picture", "")  # Safely get picture URL
                if profile_picture:  
                    st.image(profile_picture, width=40)
                else:  
                    st.image("user.png", width=40)  # Use a default placeholder
            with col2:
                st.write(f"**{user['name']}**")
                st.write(f"📧 {user['email']}")
                    # Feedback Section

            st.subheader("💬 Feedback")
            feedback = st.text_area("Let us know your thoughts!", key="feedback_input")

            if st.button("Submit Feedback"):
                add_feedback(user["sub"], feedback)
                st.success("Thank you for your feedback!")

            if st.button("Logout", use_container_width=True):
                st.session_state.pop("google_user", None)
                st.rerun()

        st.divider()

        if "google_user" in st.session_state:
            user_info = st.session_state["google_user"]

            if st.button("➕ New Chat", use_container_width=True):
                st.session_state["current_chat_history"] = []
                st.session_state.pop("current_chat_id", None)
                st.rerun()

            st.subheader("📝 Chat History")
            for chat in user_info["chat_history"]:
                chat_title = chat["title"]
                chat_id = chat["chat_id"]
                if st.button(f"{chat_title}", key=chat_id):
                    load_chat(user_info["sub"], chat_id)
                    clicked_previous_chat = True
                    st.session_state["current_chat_id"] = chat_id

    # ==== MAIN CONTENT ====
    st.title("🌱 CLIP Crop & Disease Detection")
    st.write("Upload an image and let AI detect potential crop diseases and provide insights.")
    
    selected_crop = st.selectbox("Select uploaded crop type for better accuracy", ["Select a Crop", "Wheat", "Rice", "Corn", "Potato"], 
        key="crop_selection_popup", help="Select a crop or use all available classes.")
    custom_captions = get_classes(selected_crop)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="unique_key_1")

    if uploaded_file is not None:
        # Load the fine-tuned CLIP model
        model, preprocess, device = load_custom_clip_model()
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)
        best_caption, confidence = classify_image(image, model, preprocess, device, custom_captions)
        # st.write("Prediction:" , best_caption, "Confidence:", confidence) 

        # Generate AI response if no history exists (prevents from regenerating description on each rerun)
        if not st.session_state["current_chat_history"]:
            # Reset chat history if it exists (User switched crop type after uploading image)
            if st.session_state["current_chat_history"]:
                st.session_state["current_chat_history"] = []
            clip_description = generate_clip_description(best_caption, confidence, language)
            print("Clip Description: ", clip_description)
            if user is not None:
                if "current_chat_id" not in st.session_state:
                    title = generate_chat_title(clip_description, language)
                    st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
                update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], clip_description)

    # Chat Interaction
    st.subheader("💬 Chat with LLAMA")
    if clicked_previous_chat:
        display_current_chat()

    user_prompt = st.chat_input("Ask LLAMA about this diagnosis...")
    if user_prompt:
        messages = process_user_input(user_prompt, language)  # Handle user query
        if user is not None:
            if "current_chat_id" not in st.session_state:
                title = generate_chat_title(user_prompt, language)
                st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], messages[0])
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], messages[1])
        prompts += 1

if __name__ == "__main__":
    main()
