import functions_framework
from google.cloud import bigquery
from openai import OpenAI
from datetime import datetime
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize BigQuery client
gbq_client = bigquery.Client()

# Define the cost per token for different models
COST_PER_TOKEN = {
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},  # $5.00 per 1M input tokens, $15.00 per 1M output tokens
    "gpt-3.5-turbo-0125": {"input": 0.0005 / 1000, "output": 0.0015 / 1000}  # $0.50 per 1M input tokens, $1.50 per 1M output tokens
}

@functions_framework.http
def update_monthly_run_costs(request):
    current_month = datetime.utcnow().strftime('%Y-%m')
    table_id = "chat-cache.actualism_chatbot.chatbot_runs"

    # Query to get all rows for the current month
    query = f"""
        SELECT thread_id, run_id
        FROM `{table_id}`
        WHERE FORMAT_TIMESTAMP('%Y-%m', TIMESTAMP(timestamp)) = '{current_month}'
        AND cost = 0
    """
    query_job = gbq_client.query(query)
    rows = query_job.result()

    for row in rows:
        thread_id = row.thread_id
        run_id = row.run_id

        # Retrieve run details from OpenAI API
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        prompt_tokens = run['usage']['prompt_tokens']
        completion_tokens = run['usage']['completion_tokens']
        model = run['model']

        # Calculate the cost
        input_cost = prompt_tokens * COST_PER_TOKEN.get(model, {}).get("input", 0)
        output_cost = completion_tokens * COST_PER_TOKEN.get(model, {}).get("output", 0)
        total_cost = input_cost + output_cost

        # Update the cost in BigQuery
        update_query = f"""
            UPDATE `{table_id}`
            SET cost = {total_cost}
            WHERE thread_id = '{thread_id}' AND run_id = '{run_id}'
        """
        update_job = gbq_client.query(update_query)
        update_job.result()  # Wait for the job to complete

    return "Monthly run costs updated successfully", 200