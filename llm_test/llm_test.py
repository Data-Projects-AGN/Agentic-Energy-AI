import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(base_url=os.environ["JETSTREAM_BASE"],
                api_key=os.environ["JETSTREAM_API_KEY"])

resp = client.chat.completions.create(
    model=os.environ["JETSTREAM_MODEL"],
    messages=[{"role":"user","content":"Hi"}],
    max_tokens=128, temperature=0.2
)
print(resp.choices[0].message.content)
