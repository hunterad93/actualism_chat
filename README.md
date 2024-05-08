# actualism_chat
a fun chatbot that uses 'actualfreedom.com.au' as its knowledge base for retrieval augmented chat

A Scrapy web crawler is implemented in `scrapy_spider/` it scrapes every url on the website, and converts to a big .json file, this can then easily be converted to separate .txts for each page. RAG may work decently well on a single giant file, but it's better to have the data split up by topic.

The `vector_store_creation.ipynb` script interacts with openai to upload txt files into a knowledge base, and to create a vector database from their embeddings.

The `db_cleanup.ipynb` notebook is a quick script that cleans up the database by removing any duplicate files.

The `db_clear_last_24.ipynb` allows you to interact with the files in the openai database in a csv to easily pick which ones to delete.

The `streamlit_app/` directory contains the code for the web app that uses the assistant. It's a simple streamlit app that takes the user's input and sends it to the assistant. The assistant then returns the response and the app displays it. It includes support for 'streaming' responses which improves UI a lot. Additionally it converts the openai api response 'citations' to URLs that can be clicked on. Note that to store API keys for use in the streamlit app, you interact with the streamlit sharing GUI to add 'secrets' i.e. environment variables.

# Chat with the bot here:

https://actualismchat.streamlit.app/