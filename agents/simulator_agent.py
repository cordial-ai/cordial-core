from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai

api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

def get_llm_response( messages, is_user_visually_impaired):
    
    if(is_user_visually_impaired):
            messages =  messages + """    
            The user is suffering from color blindness. When describing objects, 
            avoid using colors as the primary descriptor. Instead, use other 
            methods to present the information. Provide additional clarification 
            to assist users who may have difficulty with visual or color-based cues.
        """
    
    messages_data = [{"role": "system", "content": "you are a helful assistant"},
                           {"role": "user", "content": messages}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k'
        messages=messages_data,
        max_tokens=None,
        n=1,
        stop=None,
        temperature=0.7,
    )
    ret = response.choices[0].message.content
    return ret
    

def generate_common_sense(game_states):
    prompt = f"""Given the following list of objects with their types, properties and the current state, please generate common sense knowledge for each object type. For each object, provide:

    Typical Function/Use: What the object is commonly used for.
    Common Location: Where the object is usually found.
    Relationships to Other Objects: How it typically interacts with or relates to other objects.
    User Interaction: How a person typically uses or interacts with the object.
    Potential Hazard: Any hazards or safety concerns associated with the current state of the object. You may refer the "drawers or doors info(ID, description, state)" column 
    of the list of objects table given below for the current state of the objects.
    Potential Hazard Description: Description of the hazard

    List of Objects(in table format):

    {game_states}

    Example Output Format:
       
    object_id: 189  
    Class: Refrigerator
    Typical Function/Use: Keeps food and beverages cold and fresh.
    Common Location: Located in the kitchen.
    Relationships to Other Objects: Stores items like fruits, vegetables, dairy products, and leftovers.
    User Interaction: Users open its door to place or retrieve food items.
    Potential Hazard: (5, middle drawer inside bottom left door, open)
    Potential Hazard Description: The middle drawer on the bottom left of the refrigerator door is open, which poses a potential hazard as it could be accidentally pulled off.
    
    Your Task:
    Please generate similar common sense knowledge, including potential hazards, for each object in the list, following the example format. These object classes are
    mandatory: Oven, Refrigerator, Washing Machine, 

    """
    
    messages_data = [{"role": "system", "content": "you are a helful assistant"},
                           {"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k'
        messages=messages_data,
        max_tokens=None,
        n=1,
        stop=None,
        temperature=0.7,
    )
    ret = response.choices[0].message.content
    return ret
