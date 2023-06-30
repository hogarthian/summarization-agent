"""
Ref:
https://python.langchain.com/docs/modules/chains/popular/summarize.html
"""

import dotenv
dotenv.load_dotenv('config.env')

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain

import langchain
langchain.debug = True

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
# llm = OpenAI(temperature=0)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)


# chain = load_summarize_chain(llm, chain_type="map_reduce")
# chain.run(docs)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # load text from txt:
    loader = TextLoader("1.txt")
    docs = loader.load()

    doc_texts = text_splitter.split_documents(docs)

    # with open('1.txt', 'r') as f:
    #     claims = f.readlines()

    # texts = text_splitter.split_text(claims)
    #
    # from langchain.docstore.document import Document
    #
    # docs = [Document(page_content=t) for t in texts[:3]]

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    # chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
    # result = chain({"input_documents": doc_texts}, return_only_outputs=True)
    # print(result)

    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)
    result = chain({"input_documents": doc_texts}, return_only_outputs=True)
    print(result)
    # chain.run(doc_texts)

