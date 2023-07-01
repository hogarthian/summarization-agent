"""
Ref:
https://python.langchain.com/docs/modules/chains/popular/summarize.html
"""
import json
import dotenv
dotenv.load_dotenv('config.env')

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain

import langchain
langchain.debug = True

model_name = "gpt-4-0613"
test_file = "gptforall"
llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=5000)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=100)


if __name__ == '__main__':
    # load text from txt:
    loader = TextLoader(f"{test_file}.txt")
    docs = loader.load()

    doc_texts = text_splitter.split_documents(docs)

    chain = load_summarize_chain(
        llm, chain_type="refine", return_intermediate_steps=True)
    result = chain({"input_documents": doc_texts}, return_only_outputs=True)

    with open(f'{test_file}-{model_name}-refine.json', 'w') as f:
        json.dump(result, f)


