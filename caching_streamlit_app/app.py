import openai
import streamlit as st
from openai import OpenAI
import time
from pinecone import Pinecone
import datetime
import uuid
from google.oauth2 import service_account
import google.cloud.bigquery as bigquery


COST_THRESHOLD = 20
service_account_info = st.secrets["bigquery"]
credentials = service_account.Credentials.from_service_account_info(service_account_info)
gbq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gpt35_assistant_id = st.secrets["gpt35_assistant_id"]
gpt4o_assistant_id = st.secrets["gpt4o_assistant_id"]
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index(name=st.secrets["PINECONE_INDEX_NAME"])

def log_to_bigquery(client, run_id, thread_id):
    table_id = "chat-cache.actualism_chatbot.chatbot_runs"
    rows_to_insert = [{
        "run_id": run_id,
        "thread_id": thread_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "cost": 0
    }]

    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))

def get_current_months_cost(client):
    current_month = datetime.datetime.utcnow().strftime('%Y-%m')
    query = f"""
        SELECT SUM(cost) as total_cost
        FROM `chat-cache.actualism_chatbot.chatbot_runs`
        WHERE FORMAT_TIMESTAMP('%Y-%m', TIMESTAMP(timestamp)) = '{current_month}'
    """
    query_job = client.query(query)
    results = query_job.result()
    total_cost = list(results)[0].total_cost
    return total_cost if total_cost else 0

def ensure_single_thread_id():
    """
    Creates a thread_id and stores in session state, unless it already exists in which case it retrieves the id.
    """
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
    return st.session_state.thread_id

def get_filename(file_id):
    """
    OpenAI returns file ids with their annotations, filenames can be looked up via file id. This
    function takes an id and returns the name.
    """
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
    """
    This function converts filenames into URLs, it is specifically geared towards how I saved the webpages into txts after scraping.
    """
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

def stream_generator(prompt, thread_id, assistant_id, max_prompt_tokens=20000):
    """
    This function creates the streaming effect, but in markdown. Streamlit has a built in streaming effect but it seemed
    to fail to properly display certain markdown responses that the LLM intended like nice numbered lists. The function
    receives a prompt and a thread_id and yields partial responses as they come in from the API. Additionally it calls the
    citation handling functions.
    """
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
            max_prompt_tokens=max_prompt_tokens          
        )
        partial_response = ""
        for event in stream:
            if event.event == "thread.run.queued":
                run_id = event.data.id
                log_to_bigquery(gbq_client, run_id, thread_id)
                print(f"Run ID: {run_id}")
            elif event.data.object == "thread.message.delta":
                for content in event.data.delta.content:
                    if content.type == 'text':
                        text_value = content.text.value
                        annotations = content.text.annotations
                        if annotations:
                            for annotation in annotations:
                                citation_info = format_citation(annotation)
                                # indexes = f"from index {annotation.start_index} to {annotation.end_index}]"
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
    """
    Takes a prompt and response (only triggers for initial conversational turn) and upserts a vector record to pinecone db.
    It utilizes uuid to create an identifier and datetime to create a datetime.
    #TODO add cached response user rating (thumbs up vs thumbs down)
    #TODO add a list of file ids referenced.
    """
    unique_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    embedding = to_dense_vector_openAI(prompt, client)
    metadata = {
        "prompt": prompt,
        "response": response,
        "timestamp": timestamp,
        "unique_id": unique_id,
        "rating": 0
    }
    # Ensure the upsert data structure matches the working example
    upsert_data = {
        "id": unique_id,
        "values": embedding,  # Assuming 'values' is the correct field for embeddings
        "metadata": metadata
    }
    # Perform the upsert operation
    index.upsert(vectors=[upsert_data])  # Adjust according to the actual method signature if needed
    return None

def retrieve_cached_response_from_pinecone(prompt):
    """
    Embed a prompt and check for cached responses to similar prompts. Similarity threshold can be changed.
    Higher similarity threshold (.9+) means you will only get a cached response for virtually the same question.
    Lower similarity threshold decreases performance (higher potential for irrelevant answer) and cost.
    """
    # Embed the prompt using OpenAI's embedding API
    query_embedding = to_dense_vector_openAI(prompt, client)
    
    # Perform the query using the new Pinecone syntax
    query_response = index.query(
        namespace="", 
        top_k=1,
        vector=query_embedding,
        include_metadata=True
    )
    if query_response and query_response['matches']:
        print(query_response)
        # Check if the top result meets a similarity threshold
        if query_response['matches'][0]['score'] > 0.88: 
            print(query_response)
            return query_response['matches'][0]['metadata']
    return None

def update_rating_in_pinecone(unique_id, increment):
    """
    Fetch the current metadata, update the rating, and upsert back to Pinecone.
    """
    print("updating")
    current_data = index.fetch(ids=[unique_id])
    if current_data and 'vectors' in current_data and unique_id in current_data['vectors']:
        current_vector = current_data['vectors'][unique_id]
        current_metadata = current_vector['metadata']
        current_rating = current_metadata.get('rating', 0)
        updated_rating = current_rating + increment
        current_metadata['rating'] = updated_rating
        updated_data = {
            "id": unique_id,
            "values": current_vector['values'],  # Reuse existing embedding
            "metadata": current_metadata
        }
        print(f"Updating rating for {unique_id}: {current_rating} -> {updated_rating}")
        index.upsert(vectors=[updated_data])
    else:
        print(f"No data found for ID {unique_id}")

def like_action():
    print("liked")
    update_rating_in_pinecone(unique_id, 1)  # Call update function with "liked" status

def dislike_action():
    print('disliked')
    update_rating_in_pinecone(unique_id, -1)  # Call update function with "disliked" status

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
        st.write(prompt)  # Display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})  # Append user message to session state

    # Check for cached response first
    cached_response = retrieve_cached_response_from_pinecone(prompt)

    if cached_response:
        response_text = cached_response['response']
        unique_id = cached_response['unique_id']
        rating = cached_response.get('rating', 0)  # Retrieve the rating from the metadata
        cached_indicator = f"🗂️ Cached Response to: '{cached_response['prompt']}' (rating: {rating})\n---\n"
        full_response_text = f"{cached_indicator}{response_text}"
        with st.chat_message("assistant"):
            st.markdown("🌺 " + full_response_text)

        col1, col2, col3, col4 = st.columns([3, 3, 0.5, 0.5])

        with col3:
            st.button("👍", key="like", on_click=like_action)

        with col4:
            st.button("👎", key="dislike", on_click=dislike_action)

    else:
        thread_id = ensure_single_thread_id()
        with st.chat_message("assistant"):
            response_container = st.empty()  # Create an empty container for the response
            full_response = ""
            current_month_cost = get_current_months_cost(gbq_client)
            if current_month_cost > COST_THRESHOLD:
                assistant_id = gpt35_assistant_id
            else:
                assistant_id = gpt4o_assistant_id
            for chunk in stream_generator(prompt, thread_id, assistant_id):
                full_response += chunk
                # Update the container with the latest full response
                response_container.markdown("🌺 " + full_response)
        response_text = full_response

        if len(st.session_state.messages) <= 2:  # Assuming the first user prompt and the first assistant response
            upload_to_pinecone(prompt, response_text)
        full_response_text = response_text

    st.session_state.messages.append({"role": "assistant", "content": full_response_text})