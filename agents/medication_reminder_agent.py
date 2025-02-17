from typing import Literal
from flask import jsonify
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from datetime import datetime
from langchain_openai import ChatOpenAI
import os

from util import get_current_persona

# Static mapping for time of day phrases to actual times
TIME_OF_DAY_MAPPING = {
    "morning": "08:00 AM",
    "afternoon": "12:00 PM",
    "evening": "06:00 PM",
    "night": "08:00 PM"
}

# Initialize the ChatOpenAI model once
MODEL = ChatOpenAI(model="gpt-4o-mini")


def create_medication_reminder_agent_app(cache):
    def get_profile_data():
        # Try fetching data from cache
        cached_data = cache.get('my_cached_data')
        if cached_data is not None:
            return cached_data #jsonify({"data": cached_data, "source": "cache"})

        data = get_current_persona()
        try:
            cache.set('my_cached_data', data)  # Store in cache
        except Exception as e:
            print(e)
        return data #jsonify({"data": data, "source": "database"})

    # Helper function to check and update medication status
    def check_and_update_medication_core(medication: str, time: str) -> str:
        user_context = get_profile_data()
        # user_context = user_context.get("data")
        user_context = user_context.get("medication")
        medication_schedule = user_context.get('medication_schedule', {})
        medication_aliases = user_context.get('medication_aliases', {})
        
        if medication in medication_schedule:
            print("inside if condition")
            for dose in medication_schedule[medication]:
                if dose["status"] == "taken":
                    return f"{medication} already taken at {time}."
                else:
                    dose["status"] = "taken"
                    return f"{medication} has been taken at {time}."
        print(f"No such medication or incorrect time for {medication}.")
        return f"No such medication or incorrect time for {medication}."

    @tool
    def check_and_update_medication_by_alias_or_name(medication: str, time: str) -> str:
        """
        Use this tool when the user indicates they have taken a medication.
        It checks if the specified medication (using either its name or alias) has been taken at the specified time, and updates its status if not.

        Parameters:
        - medication: The name or alias of the medication.
        - time: The time the medication was taken (e.g., 'morning', '08:00 AM').

        Returns:
        - A message indicating whether the medication was already taken or has been updated.
        """
        # Logging for debugging
        print(f"Tool Invoked: check_and_update_medication_by_alias_or_name(medication={medication}, time={time})")
        user_context = get_profile_data()
        # user_context = user_context.get("data")
        user_context = user_context.get("medication")
        medication_schedule = user_context.get('medication_schedule', {})
        medication_aliases = user_context.get('medication_aliases', {})

        medication_name = medication_aliases.get(medication.lower(), medication)
        mapped_time = TIME_OF_DAY_MAPPING.get(time.lower(), time)
        return check_and_update_medication_core(medication_name, mapped_time)

    @tool
    def get_medication_status() -> str:
        """
        Use this tool to get the current status of all medications in the user's schedule.

        Returns:
        - A message listing the status of each medication (taken or not taken).
        """
        # Logging for debugging
        print("Tool Invoked: get_medication_status()")
        user_context = get_profile_data()
        user_context = user_context.get("medication")
        try:
            medication_schedule = user_context.get('medication_schedule', {})
        except Exception as e:
            print(e)

        status_list = []
        for med, doses in medication_schedule.items():
            med_status = f"{med.capitalize()}:"
            for dose in doses:
                med_status += f"\n  - {dose['time']}: {dose['status'].capitalize()}"
            status_list.append(med_status)
            
        print(status_list)
        return "\n".join(status_list)

    @tool
    def get_current_time() -> str:
        """
        Use this tool to get the current time in the format HH:MM AM/PM.

        Returns:
        - The current time as a string.
        """
        # Logging for debugging
        current_time = datetime.now().strftime('%I:%M %p')
        print(f"Tool Invoked: get_current_time() -> {current_time}")
        return current_time

    # Define the tools for the agent to use
    tools = [
        check_and_update_medication_by_alias_or_name,
        get_medication_status,
        get_current_time
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
        system_prompt = (
            "You are an elder-care agent responsible for managing medication intake.\n"
            "Your job is to track and remind the user about their medications.\n"
            "You have access to the following tools:\n"
            "- get_medication_status: Use this to get the status of all medications.\n"
            "- get_current_time: Use this to get the current time.\n"
            "\n"
            "Agent Policy:\n"
            "1. Always use the tools to update or retrieve medication information.\n"
            "2. If the user mentions taking a medication, use the 'check_and_update_medication_by_alias_or_name' tool to update its status.\n"
            "3. Confirm the action to the user after updating.\n"
            "4. Be polite and supportive in your responses.\n"
            "5. When you need information not covered by the tools, use the 'get_current_time' tool to fetch the current time.\n"
            "6. After providing medication-related information, check if the user understands. For example: 'Does that make sense?' or 'Would you like me to explain that again?'\n"
            "7. Always refer to medications using clear and consistent terminology. Avoid ambiguous references like 'blue pill' or 'your usual tablet.' Use personalized aliases or specific names when available.\n"
            "8. Never ask two questions at once. If you have multiple questions, break them into separate interactions to avoid overwhelming the user.\n"
            "9. When discussing medications, explain their purpose and benefits in simple terms. Avoid vague responses, and check if the user needs more information or clarification.\n"
            "10. If the user expresses concern about side effects or reluctance, respond empathetically. For example: \n"
            "    - 'I understand it might be worrying, but this pill helps control your blood pressure to prevent serious health issues. Let me know if you'd like more details.'\n"
            "    - 'If you have concerns about side effects, we can discuss them together.'\n"
            "11. Avoid clinical phrases like 'It's important for maintaining your overall health.' Instead, use a conversational tone. For example: \n"
            "    - 'This helps keep your heart healthy and strong, so it’s important to take it every day.'\n"
            "    - 'Taking this regularly can help you feel better and avoid complications later.'\n"
            "12. Always address the user in a warm, engaging way, acknowledging their efforts or feelings. For example: \n"
            "    - 'You’re doing great in managing your health! How are you feeling about this?'\n"
            "    - 'Let’s make sure everything is working well for you today.'\n"
            "\n"
            "Please make sure to follow these policies strictly and respond in a spoken style, not in a written format. Use the format below: \n"
            'speak(" your message here ")'
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
