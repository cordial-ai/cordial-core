from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain import PromptTemplate
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Load the ethics PDF
pdf_path = "https://www.skillsforcare.org.uk/resources/documents/Support-for-leaders-and-managers/Managing-people/Code-of-conduct/Code-of-Conduct.pdf"
loader = PyPDFLoader(pdf_path)
ethics_documents = loader.load()

# Extract the content (you can store it in a variable)
ethics_text = " ".join([doc.page_content for doc in ethics_documents])


# Define a prompt template that applies ethical guidance
ethics_prompt_template = """
You are an elder care robot in an elder care home in Aberdeen, UK, capable of answering questions by calling tools. According to the following ethical guidelines:
{ethics_guidelines}

Please review the following response to ensure it aligns with these ethical principles while maintaining the original style, warmth, and natural conversational tone of the response.

Original Response: "{response}"

If the response already adheres to the guidelines, repeat it as it is. If adjustments are needed, make only minimal changes to ensure ethical compliance while preserving the friendly, comforting, and supportive tone. Use British English.

Only provide the updated response.

"""

# Create the template using LangChain's PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["ethics_guidelines", "response"],
    template=ethics_prompt_template
)


# Initialize the language model
# llm = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define the function for ethical post-processing
def apply_ethics_after_response(response):
    # Create the prompt for reviewing the response according to ethics
    prompt = prompt_template.format(ethics_guidelines=ethics_text, response=response)
    
    # Generate the ethically adjusted response or approval
    ethical_response = llm(prompt)
    # st.text(ethical_response)
    cleaned_content = ethical_response.content
    # cleaned_content = ethical_response.replace("Original Response:", "").strip()
    
    # Remove first and last double quotes if they exist
    if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
        cleaned_content = cleaned_content[1:-1]
    else:
        cleaned_content = cleaned_content
        
    return cleaned_content
