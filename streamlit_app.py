import streamlit as st
from PIL import Image
import numpy as np

# Importing CLIP and LLaMA utilities
from src.clip_utils import load_clip_model, get_text_features, get_image_features, compute_similarity, load_custom_clip_model
from src.classes import get_candidate_captions
from src.oauth import get_login_url, get_google_info
from src.firebase_config import create_user_if_not_exists, create_new_chat, fetch_chat_history, load_chat, update_chat_history
from src.llama_utils import generate_clip_description, process_user_input, display_current_chat, generate_chat_title


def main():
    st.set_page_config(page_title="CLIP Crop & Disease Detection", layout="wide")

    # Load models
    model, preprocess, device = load_clip_model()
    candidate_captions = get_candidate_captions()
    clicked_previous_chat = False # temporary.
    prompts = 0 # Track # of prompts. Not ideal method

    # Initialize session state
    if "current_chat_history" not in st.session_state:
        st.session_state["current_chat_history"] = []
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    # ==== SIDEBAR (Chat History + Google Sign-In) ====
    with st.sidebar:
        st.title("Account")
        user = None
        # Google Sign-In Section
        if "google_user" not in st.session_state:
            login_url = get_login_url()
            st.markdown(f'<a href="{login_url}" target="_self"><button style="width: 100%;">🔑 Sign in with Google</button></a>', unsafe_allow_html=True)

            query_params = st.query_params
            # Check if user has attempted to log in ('code' in url)
            if "code" in query_params:
                auth_code = query_params["code"]
                user_info = get_google_info(auth_code)
                if user_info:
                    user_id = user_info["sub"]
                    # Create user entry in Firestore if not exists
                    create_user_if_not_exists(user_id, user_info["name"], user_info["email"])
                    # Set user info in session state
                    user_info["chat_history"] = fetch_chat_history(user_id)
                    st.session_state["google_user"] = user_info
                
                    st.rerun()
        # If logged in, display user info and logout button
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
            col1, col2 = st.columns([1, 4])  # Align profile picture and text
            with col1:
                st.image(user["picture"], width=40)
            with col2:
                st.write(f"**{user['name']}**")
                st.write(f"📧 {user['email']}")

            if st.button("Logout", use_container_width=True):
                st.session_state.pop("google_user", None)
                st.rerun()
                        # Inject custom CSS to style the chat history buttons
        st.divider()

        st.subheader("📝 Chat History")
        if "google_user" in st.session_state:
            user_info = st.session_state["google_user"]
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
        if not st.session_state["current_chat_history"]:
            clip_description = generate_clip_description(best_caption, confidence)
            print("Clip Description: ", clip_description)
            if "current_chat_id" not in st.session_state:
                title = generate_chat_title(clip_description)
                st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], clip_description)

    # Chat Interaction
    st.subheader("💬 Chat with LLAMA")
    if clicked_previous_chat:
        display_current_chat()
    user_prompt = st.chat_input("Ask LLAMA about this diagnosis...")
    if user_prompt:
        messages = process_user_input(user_prompt)  # Handle user query
        if user:
            # Create title after responding so user doesnt wait for 2 responses
            if "current_chat_id" not in st.session_state:
                title = generate_chat_title(user_prompt)
                st.session_state["current_chat_id"] = create_new_chat(user["sub"], title)
        
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], messages[0])
            update_chat_history(st.session_state["google_user"]["sub"], st.session_state["current_chat_id"], messages[1])
        prompts += 1

if __name__ == "__main__":
    main()