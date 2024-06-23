import os
from pinecone import Pinecone
from openai import OpenAI
from anthropic import Anthropic
import cohere

# Initialize clients
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index("actualism-website")
openai_client = OpenAI()
claude_client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
cohere_client = cohere.Client(os.environ.get('COHERE_API_KEY'))

def rephrase_query_for_similarity_search(query):
    messages = [
        {"role": "system", "content": "You are an assistant who responds to queries by providing a detailed hypothetical answer (even if its a guess)."},
        {"role": "user", "content": query}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        max_tokens=300,
        temperature=0
    )
    return response.choices[0].message.content

def generate_embeddings(query):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        encoding_format="float",
        dimensions=1024
    ).data[0].embedding

def search_query(query):
    vector = generate_embeddings(query)
    return index.query(
        namespace="",
        top_k=100,
        vector=vector,
        include_metadata=True
    )

def rerank_results(query, matches):
    docs = [match['metadata']['raw_string'] for match in matches]
    response = cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=5
    )
    return response

def filter_metadata_by_relevance(metadata_list, score_threshold=0.1):
    filtered_metadata = [metadata for metadata in metadata_list if metadata['relevance_score'] >= score_threshold]
    if not filtered_metadata:
        filtered_metadata.append(max(metadata_list, key=lambda x: x['relevance_score']))
    return filtered_metadata

def calculate_confidence_score(metadata_list):
    if not metadata_list:
        return 0
    sum_relevance = sum(item['relevance_score'] for item in metadata_list)
    average_relevance = sum_relevance / len(metadata_list)
    num_documents = len(metadata_list)
    return (sum_relevance * average_relevance) / num_documents

def generate_claude_response(query, metadata_list):
    formatted_results = "\n\n".join([f"Document: {result['raw_string']}" for result in metadata_list])
    prompt = f"Using the search results provided: <search_results>{formatted_results}</search_results>, please answer the following question <question>{query}</question>."
    response = claude_client.messages.create(
        model="claude-3-sonnet-20240229",
        system="Use only the information from the search results when responding.",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )
    return response.content[0].text

def main(query):
    rephrased_query = rephrase_query_for_similarity_search(query)
    results = search_query(query)
    additional_results = search_query(rephrased_query)
    results['matches'].extend(additional_results['matches'])

    reranked_results = rerank_results(query, results['matches'])
    selected_metadata = []
    for item in reranked_results.results:
        metadata = results['matches'][item.index]['metadata']
        metadata['relevance_score'] = item.relevance_score
        selected_metadata.append(metadata)

    filtered_metadata = filter_metadata_by_relevance(selected_metadata)
    confidence_score = calculate_confidence_score(filtered_metadata)
    response = generate_claude_response(query, filtered_metadata)

    return response, confidence_score

if __name__ == "__main__":
    query = "How can I get started trying to become virtually free? What is the first thing to do?"
    response, confidence = main(query)
    print(f"Response: {response}")
    print(f"Confidence Score: {confidence}")