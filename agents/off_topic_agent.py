from typing import Literal
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


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


from util import get_current_persona


# Initialize the ChatOpenAI model once
MODEL = ChatOpenAI(model="gpt-4o-mini")

def create_off_topic_agent_app(cache):
    def get_profile_data():
        cached_data = cache.get('my_cached_data')
        if cached_data is not None:
            return cached_data #jsonify({"data": cached_data, "source": "cache"})

        data = get_current_persona()
        try:
            cache.set('my_cached_data', data)  # Store in cache
        except Exception as e:
            print(e)
        return data 

    def get_weather(city):
        api_key = os.getenv("WEATHER_API_KEY")
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = base_url + "appid=" + api_key + "&q=" + city + "&units=metric"
        response = requests.get(complete_url)
        data = response.json()
        return data

    def get_uk_news():
        # Parse the RSS feed using feedparser
        bbcUKNewsPolitics = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/uk_politics/rss.xml")
        bbcUKNewsScotland = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/scotland/rss.xml")
        bbcUKNewsEngland = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/england/rss.xml")
        bbcUKNewsBusiness = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/business/rss.xml")
        bbcUKNewsHealth = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/world/rss.xml")
        bbcUKNewsWorld = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/health/rss.xml")
        bbcUKNewsWales = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/wales/rss.xml")
        bbcUKNewsFrontPage = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml")
        bbcUKNewsTechnology = feedparser.parse("http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/technology/rss.xml")

        docs = []
        for entry in bbcUKNewsPolitics.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsScotland.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsEngland.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsBusiness.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsHealth.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsWorld.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsWales.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsFrontPage.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        for entry in bbcUKNewsTechnology.entries:
            content = f"{entry.title}\n\n{entry.summary}"
            docs.append(content)
        return docs

    def get_current_time():
        # Define the UK timezone
        uk_timezone = pytz.timezone("Europe/London")
        # Get the current time in the UK
        uk_time = datetime.now(uk_timezone)
        return(uk_time.strftime("%Y-%m-%d %H:%M:%S"))


    # Load your language model (OpenAI GPT-based in this case)
    # llm = OpenAI(temperature=0.5)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Define tools for the agent
    # Tool for weather information
    @tool
    def weather_tool(query: str) -> str:
        """Provides weather information for a specific city"""
        city = query.split("in")[-1].strip()
        return get_weather(city)

    @tool
    def current_time_tool(query: str) -> str:
        """Provides current time"""
        return get_current_time()

    # Tool for news
    @tool
    def news_tool(query: str) -> str:
        """Fetches the latest news."""
        return get_uk_news()

    # Define the tool for general knowledge using ChatGPT
    @tool
    def knowledge_tool(query: str) -> str:
        """Provides general knowledge on predefined topics using ChatGPT."""
        # Use ChatGPT to answer the query
        response = llm.invoke([{"role": "system", "content": """You are an elder care robot in an elder care home in Aberdeen, UK and you are an expert in general knowledge.
        Elders love to talk about anything. Dont show off that you know more. You need to be more friendly but respectful. """},
                            {"role": "user", "content": query}])
        # return response["content"]
        return response.content

    # Define the tools for the agent to use
    tools = [
        weather_tool,
        current_time_tool,
        news_tool,
        knowledge_tool
    ]

    tool_node = ToolNode(tools)

    # Bind the model with tools
    model = MODEL.bind_tools(tools)

    # Define the function that determines whether to continue or not
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Define the function that calls the model
    def call_model(state: MessagesState):
        user_context = get_profile_data()
        user_context = user_context.get("persona_description")

        system_prompt = (
            """You are an elder care robot in an elder care home in Aberdeen, UK, capable of answering questions by calling tools. 
        You have access to recent news around the world. Elders love to talk about news, politics, and gossip. You need to be more 
        friendly but respectful. Now act like an elder care robot. Your responses should be limited to 3-4 sentences max per response. Your conversation should be more casual than formal.
        If appropriate start the response with empathy acknowledgement of the user's input.

        If the user greets morning, ask whether the user had breakfast
        If the user greets afternoon, ask whether the user had lunch
        If the user greets evening or night, ask whether the user had dinner

        Never ask two questions at once.
        The elder will initiate the conversation.

        When the user asks a question, think step by step to determine which tool you should call.
        You should call a tool only 1 time and get the response of that tool. 
        If a tool call is necessary, call it. Then return the final answer to the user.
                
        Following is user's information        
        """ 
        + user_context

        )
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        response = model.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", 'agent')

    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)

    return app
