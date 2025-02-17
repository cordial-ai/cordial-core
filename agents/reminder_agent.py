import feedparser
import requests
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.memory import ConversationBufferMemory
from datetime import datetime
import pytz
import os

def get_current_time():
    # Define the UK timezone
    uk_timezone = pytz.timezone("Europe/London")
    # Get the current time in the UK
    uk_time = datetime.now(uk_timezone)
    # Print the current time in the UK
    return(uk_time.strftime("%Y-%m-%d %H:%M:%S"))


# Load your language model (OpenAI GPT-based in this case)
# llm = OpenAI(temperature=0.5)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define tools for the agent
# Tool for weather information
@tool
def current_time_tool(query: str) -> str:
    """Provides current time"""
    return get_current_time()


def create_reminder_agent(user_context):
    
    # Create LangChain Tool objects
    tools = [ current_time_tool]

    # Define a prompt template for the ZeroShotAgent
    prompt_def = """You are an elder care robot in an elder care home in Aberdeen, UK, capable of answering questions by calling tools. 
    You have access to recent news around the world. Elders love to talk about news, politics, and gossip. You need to be more 
    friendly but respectful. Now act like an elder care robot. Your responses should be limited to 3-4 sentences max per response. Your conversation should be more casual than formal.
    The elder will initiate the conversation.

    When the user asks a question, think step by step to determine which tool you should call. 
    If a tool call is necessary, call it. Then return the final answer to the user.
    """ 
    
    
    prompt_head = prompt_def + user_context
    prompt_template = prompt_head + """

    The query from the user is: {input}
    
    Chat History:
    {chat_history}
    
    Intermediate actions and thoughts:
    {agent_scratchpad}"""

    # prompt = PromptTemplate(template=prompt_template)
    prompt = PromptTemplate(input_variables=["input", "chat_history"], template=prompt_template)

    # Construct the tool calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Define the memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    
    return agent_executor
