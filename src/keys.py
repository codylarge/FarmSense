import os

# Files:
# - ./groq_api_key.txt
# - ./openai_api_key.txt
# Each file must be one line with only the api key


# load api keys from text file 
def load_api_key(file_path, env_var_name):
    with open(file_path, "r") as file:
        os.environ[env_var_name] = file.read().strip()