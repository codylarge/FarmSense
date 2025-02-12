from google_auth_oauthlib.flow import Flow
import os
import pickle
import pathlib
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError

from google.oauth2 import id_token
from google.auth.transport import requests

# Google OAuth Credentials
CLIENT_SECRETS_FILE = "client_secret.json" 
SCOPES = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
REDIRECT_URI = "http://localhost:8501"

def get_login_url():
    """Generate Google OAuth login URL"""
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return auth_url

def get_user_info(auth_code):
    """Exchange auth code for user credentials and fetch user info"""
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=auth_code)
    credentials = flow.credentials

    # Verify and decode the ID token
    try:
        id_info = id_token.verify_oauth2_token(credentials.id_token, requests.Request())

        # Extract user information safely
        user_info = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "id_token": credentials.id_token,
            "email": id_info.get("email", "N/A"),
            "name": id_info.get("name", "N/A"),
            "picture": id_info.get("picture", ""),
        }
        return user_info

    except ValueError as e:
        st.error(f"Error verifying ID token: {e}")
        return None