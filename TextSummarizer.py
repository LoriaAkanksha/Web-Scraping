import pandas as pd
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
from groq import Groq
from dotenv import load_dotenv
import warnings
import os

warnings.filterwarnings("ignore")
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

url1 = "https://docs.tavily.com/docs/gpt-researcher/pip-package"
loader = AsyncChromiumLoader([url1])
data = loader.load()

tt = Html2TextTransformer()
docs = tt.transform_documents(loader.load())
ts = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
fd = ts.split_documents(docs)

summary = ""

for xx in fd:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "you are a helpful intelligent assistant"},
            {
                "role": "user",
                "content": f"summarize the following into bullet points, only consider meaningful sentences, also ignore all headings and words:\n\n{xx}",
            },
        ],
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=512,  # Adjust this to aim for around 50% of the original words
        top_p=1,
        stop=None,
        stream=False,
    )
    print("-----------------------------------------------------------------------------------------------------")
    print(chat_completion.choices[0].message.content)
    summary += chat_completion.choices[0].message.content + "\n"

doc = Document()
doc.add_paragraph(summary)
doc.save("output.docx")

print("Document saved successfully.")


