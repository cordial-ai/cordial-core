from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import os
import streamlit as st


# Example dialogues for off-topic and medical categories
off_topics = [
    "What is your favorite color?",
    "Tell me about your weekend plans.",
    "Tell me about news",
    "what are the latest news?"
    "Tell me about weather",
    "Good Morning",
    "Good Afternoon",
    "Good Evening",

]

medical_dialogues = [
    "What is the treatment for diabetes?",
    "Can you explain the symptoms of high blood pressure?",
    "What medications are used to treat asthma?",
    "Remind me my medication",
    "When I need to taje medications",
    "I want to take my blue pill",
    "What is Bisoprolol?"
]

simulator_dialogues = [
    "Go to kitchen",
    "Come here",
    "Can you please go and collect my newspaper",
    "Bring me a book",
    "Bring me an orange",
]

embeddings = OpenAIEmbeddings()

off_topic_embeddings = embeddings.embed_documents(off_topics)
medical_dialogue_embeddings = embeddings.embed_documents(medical_dialogues)
simulator_dialogue_embeddings = embeddings.embed_documents(simulator_dialogues)

def query_router(input_query, previous_agent="INIT", chat_source="SIM"):
    if(chat_source=="SIM"):
        player_text = input_query.split("Please output your action now.")[1].split("Agent:")[0].strip()
    else:
        player_text = input_query
    query_embedding = embeddings.embed_query(player_text)

    # Compute Euclidean distances and take the mean for each comparison
    off_topic_distance = pairwise_distances([query_embedding], off_topic_embeddings, metric='euclidean').mean()
    medical_distance = pairwise_distances([query_embedding], medical_dialogue_embeddings, metric='euclidean').mean()
    simulator_distance = pairwise_distances([query_embedding], simulator_dialogue_embeddings, metric='euclidean').mean()

    off_topic_similarity = 1 / (1 + off_topic_distance)  # The +1 prevents division by zero
    medical_similarity = 1 / (1 + medical_distance)
    simulator_similarity = 1 / (1 + simulator_distance)

    print("off_topic_similarity: " + str(off_topic_similarity))
    print("medical_similarity: " + str(medical_similarity))
    print("simulator_similarity: " + str(simulator_similarity))
    print("previous_agent: " + str(previous_agent))

    threshold_topic = 0.01
    if(abs(abs(medical_similarity - off_topic_similarity) - simulator_similarity) < threshold_topic and previous_agent != "INIT"):
        return previous_agent
    else:
        highest_similarity = max(medical_similarity, off_topic_similarity, simulator_similarity)

        # Determine the category based on the highest similarity
        if highest_similarity == medical_similarity:
            return "MEDICATION_REMINDER_AGENT"
        elif highest_similarity == off_topic_similarity:
            return "OFF_TOPIC_AGENT"
        elif highest_similarity == simulator_similarity:
            return "SIMULATOR_AGENT"

        return "SIMULATOR_AGENT"  # Fallback (should not happen if scores are valid)
