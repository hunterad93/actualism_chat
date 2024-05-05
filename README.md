# actualism_chat
a fun chatbot that uses 'actualfreedom.com.au' as its knowledge base for retrieval augmented chat

Specifically it uses the downloadable 'actual freedom lite' from this page: http://www.actualfreedom.com.au/sundry/aflite.htm

The `knowledge_base_generator.ipynb` extracts all .htm files from this download, and converts them to html for compatabiity with the OpenAI vector store.

The `db_cleanup.ipynb` notebook is a quick script that cleans up the database by removing any duplicate files. This is important for maintaining the integrity of the database and ensuring that only the most recent versions of each file are stored.

The `streamlit_app/` directory contains the code for the web app that uses the assistant. It's a simple streamlit app that takes the user's input and sends it to the assistant. The assistant then returns the response and the app displays it. It includes support for 'streaming' responses which improves UI a lot.

