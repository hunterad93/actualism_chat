import openai
import streamlit as st
from openai import OpenAI
import time
from pinecone import Pinecone
import datetime
import uuid


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant_id = st.secrets["ASSISTANT_ID"]
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index('actualism-qa')


def ensure_single_thread_id():
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
    return st.session_state.thread_id

def get_filename(file_id):
    try:
        # Retrieve the file metadata from OpenAI
        file_metadata = client.files.retrieve(file_id)
        # Extract the filename from the metadata
        filename = file_metadata.filename
        return filename
    except Exception as e:
        print(f"Error retrieving file: {e}")
        return None
    
def format_citation(annotation):
    file_id = annotation.file_citation.file_id
    filename = get_filename(file_id)
    if filename:
        # Replace '---' with '/' and '.html' with '.htm' for URL conversion
        file_url = filename.replace('---', '/').replace('.txt', '')
        if not file_url.startswith('www.'):
            file_url = 'www.' + file_url
        citation_info = f" ({file_url}) "
    else:
        citation_info = "[Citation from an unknown file]"
    return citation_info

def stream_generator(prompt, thread_id):
    # Create the initial message
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    # Start streaming the response
    with st.spinner("Wait... Generating response..."):
        stream = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True,
            max_prompt_tokens=20000          
        )
        partial_response = ""
        for event in stream:
            if event.data.object == "thread.message.delta":
                print(event)
                for content in event.data.delta.content:
                    if content.type == 'text':
                        text_value = content.text.value
                        annotations = content.text.annotations
                        if annotations:
                            for annotation in annotations:
                                citation_info = format_citation(annotation)
                                indexes = f"from index {annotation.start_index} to {annotation.end_index}]"
                                text_value = f"{citation_info}"
                        partial_response += text_value
                        words = partial_response.split(' ')
                        for word in words[:-1]:  # Yield all but the last incomplete word
                            yield word + ' '
                        partial_response = words[-1]  # Keep the last part for the next chunk
            else:
                pass
        if partial_response:
            yield partial_response  # Yield any remaining text

def to_dense_vector_openAI(text, client, model='text-embedding-3-large'):
    """Generate a dense vector for a given text using OpenAI's embedding model."""
    response = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float",  # Using float for direct use in Python
        dimensions=256
    )
    # Extract the dense vector values from the response
    return response.data[0].embedding  # Assuming single text input for simplicity

def upload_to_pinecone(prompt, response):
    if len(st.session_state.messages) <= 2:  # Assuming the first user prompt and the first assistant response
        unique_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        embedding = to_dense_vector_openAI(prompt, client)
        metadata = {
            "prompt": prompt,
            "response": response,
            "timestamp": timestamp,
            "unique_id": unique_id
        }
        # Ensure the upsert data structure matches the working example
        upsert_data = {
            "id": unique_id,
            "values": embedding,  # Assuming 'values' is the correct field for embeddings
            "metadata": metadata
        }
        # Perform the upsert operation
        index.upsert(vectors=[upsert_data])  # Adjust according to the actual method signature if needed
    else:
        return None

def retrieve_cached_response_from_pinecone(prompt):
    # Embed the prompt using OpenAI's embedding API
    query_embedding = to_dense_vector_openAI(prompt, client)
    
    # Perform the query using the new Pinecone syntax
    query_response = index.query(
        namespace="",  # Adjust the namespace if needed
        top_k=1,  # You can adjust the number of top results you want to retrieve
        vector=query_embedding,
        include_metadata=True
    )
    if query_response and query_response['matches']:
        print(query_response)
        # Check if the top result meets a similarity threshold
        if query_response['matches'][0]['score'] > 0.9:  # Assuming a similarity threshold
            print(query_response)
            return query_response['matches'][0]['metadata']
    return None

# Streamlit interface
st.set_page_config(page_icon="🌺")
st.title("🌺 Discuss Actualism With ChatGPT")
st.subheader("Be wary that ChatGPT often makes mistakes and fills in the gaps with its own reasoning. Verify its responses using the provided citation links. The currently deployed version is using chat gpt 3.5 to manage costs.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your message")
# Streamlit interface
if prompt:
    with st.chat_message("user"):
        st.write(prompt)  # Display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})  # Append user message to session state

    # Check for cached response first
    cached_response = retrieve_cached_response_from_pinecone(prompt)
    if cached_response:
        response_text = cached_response['response']
        cached_indicator = "🗂️ (Cached Response)"
        full_response_text = f"{cached_indicator} {response_text}"
        with st.chat_message("assistant"):
            st.markdown("🌺 " + full_response_text)
    else:
        thread_id = ensure_single_thread_id()
        with st.chat_message("assistant"):
            response_container = st.empty()  # Create an empty container for the response
            full_response = ""
            for chunk in stream_generator(prompt, thread_id):
                full_response += chunk
                # Update the container with the latest full response, adding fire emojis
                response_container.markdown("🌺 " + full_response)
        response_text = full_response
        upload_to_pinecone(prompt, response_text)
        full_response_text = response_text

    st.session_state.messages.append({"role": "assistant", "content": full_response_text})