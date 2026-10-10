import os

from openai import OpenAI


endpoint = os.getenv(
    "NEOSTATS_BASE_URL",
    "https://neo-hackathon.services.ai.azure.com/openai/v1/",
)
deployment_name = os.getenv("NEOSTATS_MODEL", "Neo-Hackathon")
api_key = os.getenv("NEOSTATS_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Set NEOSTATS_API_KEY before running this example.")

client = OpenAI(base_url=endpoint, api_key=api_key)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
)

print(completion.choices[0].message)
