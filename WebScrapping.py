import pandas as pd
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
#from langchain_community.document_transformers import BeautifulSoupTransformer

url1 = "https://www.thehindu.com/"
loader = AsyncChromiumLoader([url1])
data = loader.load()

#print(data)
#print(data[0].page_content)

tt=Html2TextTransformer()
fd=tt.transform_documents(data)
print(fd)

'''# Create a DataFrame from the page content
df = pd.DataFrame(fd)

# Write DataFrame to CSV file
df.to_csv("pagecontent1.csv", index=False)

# Optionally, print the DataFrame
print(df)'''