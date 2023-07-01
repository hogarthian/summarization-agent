"""
Ref:
https://python.langchain.com/docs/modules/chains/popular/summarize.html
"""

import dotenv
dotenv.load_dotenv('config.env')

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import langchain
langchain.debug = True

model_name = "gpt-4-0613"
llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=7000)
# llm = OpenAI(temperature=0)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)


map_template_string = """Give the following meeting transcription, write a concise summary for this part of the meeting.
{transcription}
CONCISE SUMMARY:
"""


reduce_template_string = """Given the following meeting summary answer the following question
{summary}
Question: {question}
Answer:
"""
MAP_PROMPT = PromptTemplate(input_variables=["transcription"], template=map_template_string)
REDUCE_PROMPT = PromptTemplate(input_variables=["summary", "question"], template=reduce_template_string)

map_llm_chain = LLMChain(llm=llm, prompt=MAP_PROMPT)
reduce_llm_chain = LLMChain(llm=llm, prompt=REDUCE_PROMPT)

generative_result_reduce_chain = StuffDocumentsChain(
    llm_chain=reduce_llm_chain,
    document_variable_name="summary",
)

combine_documents = MapReduceDocumentsChain(
    llm_chain=map_llm_chain,
    combine_document_chain=generative_result_reduce_chain,
    document_variable_name="transcription",
)

if __name__ == '__main__':
    # load text from txt:
    loader = TextLoader("1.txt")
    docs = loader.load()

    doc_texts = text_splitter.split_documents(docs)

    with open('1.txt', 'r') as f:
        claims = f.readlines()
    text = "".join(claims)
    # texts = text_splitter.split_text(claims)
    #
    # from langchain.docstore.document import Document
    #
    # docs = [Document(page_content=t) for t in texts[:3]]
    map_reduce = MapReduceChain(
        combine_documents_chain=combine_documents,
        text_splitter=text_splitter,
    )

    result = map_reduce.run(input_text=text, question="What patent ideas are mentioned in the meeting?")

    # prompt_template = """Write a concise summary of the following:
    # {text}
    # CONCISE SUMMARY:"""
    # PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    # chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
    # # result = chain({"input_documents": doc_texts}, return_only_outputs=True)
    # # print(result)
    #
    # # chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)
    # result = chain({"input_documents": doc_texts}, return_only_outputs=True)

    with open(f'1-{model_name}-patent-idea-extraction.txt', 'w') as f:
        f.writelines(result)


