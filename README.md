# [Demo Video](https://www.youtube.com/watch?v=QxyiaxTg6j8&ab_channel=CodyLarge)

## [Open in Browser](https://farmsense.streamlit.app)
### Description: 
This project is a web application that helps farmers identify crop diseases using CLIP for image understanding and a Retrieval-Augmented Generation (RAG) system powered by a LLaMA-based language model. The system provides relevant answers to questions about treatment options, weed control strategies, and agricultural policies by retrieving information from a curated agricultural knowledge base. The app is designed to be accessible and helpful for farmers, especially in areas with limited access to expert support.

### How to run it on your own machine

0. Create a venv
   ```
   $ python -m venv venv
   $ venv\Scripts\activate
   ```
   
1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
