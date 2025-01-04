import os
import csv
import json
import datetime
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util
# from nltk.corpus import stopwords

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")
# nltk.download("stopwords")

# # Stopwords initialization
# stop_words = set(stopwords.words('english'))

file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
faq_model = SentenceTransformer('all-MiniLM-L6-v2')  # FAQ model
model = faq_model

# Prepare training data
tags = []
all_patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        tags.append(intent["tag"])

x = vectorizer.fit_transform(all_patterns)
y = tags
clf.fit(x, y)

# Prepare FAQ data (dummy for now, replace with actual FAQ dataset)
faq_questions = ["What is your name?", "How do you work?","What is your purpose?","What is this chatbot for?","Can this chatbot diagnose mental health conditions?","What should I do if I feel overwhelmed or in crisis?"," Can I talk to a human therapist through this chatbot?",
                "How can I manage stress effectively?","What are some quick tips to reduce anxiety?"," How do I deal with panic attacks?","Why do I feel anxious for no reason?","What are the signs of depression?","How can I feel better when I’m feeling down?",
                "What should I do if I have no motivation to do anything?","How do I talk to someone about my feelings of depression?","How can I improve my sleep quality?","Why does lack of sleep affect my mood?","What should I do if I can’t stop overthinking at night?",
                "How can I improve my relationships?","What can I do if I feel lonely?","How do I set healthy boundaries with others?","What are some self-care activities I can try?"," How can I develop a positive mindset?","What are grounding techniques, and how can they help me?",
                "How can I manage work-related stress?","What should I do if I feel burned out?","How can I improve focus and concentration?","What should I do if I feel like hurting myself?","How do I deal with a traumatic experience?","Who can I call for immediate help in a crisis?",
                "How does meditation help with mental health?"
                ]
faq_answers = ["I'm a chatbot created by Yashraj Dudhe.", "I use machine learning to respond to your queries.","My purpose is to assist you in your work.","This chatbot is designed to provide mental health support, tips, and resources. It can help you manage stress, anxiety, and other common mental health concerns, but it’s not a substitute for professional therapy or medical advice",
            "No, this chatbot cannot diagnose mental health conditions. For an accurate diagnosis, please consult a licensed mental health professional.","If you feel overwhelmed, take deep breaths and try grounding techniques. If you’re in crisis, please contact a trusted person or reach out to a helpline in your area immediately.",
            "This chatbot doesn’t connect you directly with therapists, but we can provide recommendations for licensed professionals or helplines.",
            "Try stress-management techniques like deep breathing, regular exercise, staying organized, and practicing mindfulness. Taking breaks and setting boundaries can also help.",
            "Practice deep breathing (inhale for 4 seconds, hold for 7, exhale for 8).","During a panic attack, focus on slow, deep breaths. Remind yourself that the attack will pass and you’re safe. Ground yourself by engaging your senses (e.g., holding something cold).",
            "Sometimes anxiety is triggered subconsciously or due to physical factors like lack of sleep, dehydration, or hormonal changes. Reflect on recent events or consult a professional for guidance.",
            "Common signs include persistent sadness, loss of interest in activities, fatigue, difficulty concentrating, changes in appetite or sleep, and feelings of hopelessness.",
            "Spend time with loved ones, engage in activities you enjoy, practice self-care, and seek professional help if needed.",
            "Start with small, manageable tasks. Break down larger goals into smaller steps. Reward yourself for progress and seek support from friends, family, or a therapist.",
            "Reach out to a trusted friend, family member, or mental health professional. Be honest about your feelings and ask for support.",
            "Maintain a consistent sleep schedule, create a relaxing bedtime routine, avoid screens before bed, and limit caffeine and alcohol intake.",
            "Lack of sleep can affect mood-regulating brain chemicals, leading to irritability, anxiety, and depression symptoms.",
            "Try journaling, meditation, or relaxation exercises to calm your mind. Practice mindfulness and focus on the present moment.",
            "Spend quality time together, communicate openly, show appreciation, and resolve conflicts respectfully. Seek couples therapy if needed.",
            "Join social groups, volunteer, or take up hobbies to meet new people. Reach out to friends or family for support.",
            "Communicate your needs clearly and assertively. Say no when necessary, and prioritize your well-being.",
            "Engage in activities that bring you joy, relaxation, or comfort. Practice self-compassion and prioritize your mental health.", 
            "Challenge negative thoughts, practice gratitude, and surround yourself with positive influences.",
            "Grounding techniques help you stay present and manage anxiety. Try deep breathing, progressive muscle relaxation, or focusing on your senses.",
            "Set boundaries, take breaks, and practice self-care. Communicate openly with your employer about your needs and concerns.",
            "Take time off to rest and recharge. Prioritize self-care, set boundaries, and seek support from loved ones or a therapist.",
            "Minimize distractions, break tasks into smaller steps, and take regular breaks. Practice mindfulness and avoid multitasking.",
            "If you feel like hurting yourself, please seek immediate help by contacting a crisis helpline or going to the nearest emergency room.",
            "Talk to a mental health professional or trusted person about your experience. Practice self-care, engage in relaxing activities, and seek support.",
            "In a crisis, contact a mental health helpline, emergency services, or a trusted person for immediate help.",
            "Meditation can reduce stress, improve focus, and enhance emotional well-being. It can also help you manage anxiety, depression, and other mental health concerns."
            
            ]
faq_embeddings = faq_model.encode(faq_questions, convert_to_tensor=True)


# def preprocess_text(text):
#     # Convert text to lowercase
#     text = text.lower()
#     # Tokenize text
#     words = nltk.word_tokenize(text)
#     # Remove stopwords
#     filtered_words = [word for word in words if word not in stop_words]
#     # Join words back into a single string
#     return ' '.join(filtered_words)

def chatbot(input_text):
    faq_response = faq_search(input_text)
    if faq_response:  # If a valid FAQ response is found
        return faq_response
    # preprocessed_text = input_text
    input_text_vectorized = vectorizer.transform([input_text])
    probabilities = clf.predict_proba(input_text_vectorized)[0]
    max_prob = max(probabilities)

    # Confidence threshold for fallback
    if max_prob < 0.01:
        return fallback_response(input_text)

    predicted_tag = clf.predict(input_text_vectorized)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            return response


def fallback_response(input_text):
    # FAQ search
    input_embedding = faq_model.encode(input_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(input_embedding, faq_embeddings)

    # Get the maximum score for each item in the batch (dim=0)
    max_score_idx = scores.argmax(dim=1)

    # Get the actual maximum score (assuming a single input in the batch)
    max_score = scores[0, max_score_idx[0]].item()

    if max_score > 0.7:  # Confidence threshold for FAQ match
        # Access the FAQ answer corresponding to the maximum score
        return faq_answers[max_score_idx[0].item()]

    # Default fallback response
    return "I'm sorry, I don't understand that. Can you please rephrase?"

faq_model = SentenceTransformer('all-MiniLM-L6-v2')

def faq_search(input_text):
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(input_embedding, faq_embeddings)
    max_score_idx = scores.argmax()
    if scores[0][max_score_idx] > 0.7:  # Confidence threshold
        return faq_answers[max_score_idx]

    return None
#ok good
counter = 0
conversation_history = []

def main():
    global counter
    global covenrsation_history
    st.title("CareBuddy - Mental Health Chatbot")
    
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.write("Welcome to the CareBuddy, we are always there to help you")
        
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv','w',newline = '', encoding = 'utf-8') as csvfile:
                csv_writer = csv_writer(csvfile)
                csv_writer.writehow(['User Input','Chatbot','Timestamp'])
                
        counter +=1
        user_input = st.text_input("You", key = f"user_input_(counter)")

        if user_input:
            user_id = "user"  # In a real application, this would be unique for each user
            response = chatbot(user_input)
            conversation_history.append(("You", user_input))
            conversation_history.append(("Chatbot", response))

            # Display conversation history
            for speaker, text in conversation_history:
                if speaker == "You":
                    st.markdown(f"**{speaker}:** {text}")
                else:
                    st.markdown(f"**{speaker}:** {text}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()
                
    elif choice == "Conversation History":
        st.header("Conversation History:")
            
        with open('chat_log.csv','r',encoding = 'utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("----")
                    
    elif choice == "About":

        st.subheader("About Us:")
        st.write("CareBuddy is a mental health chatbot designed to provide support, resources, and information on common mental health concerns. Our goal is to help you manage stress, anxiety, depression, and other emotional challenges in a safe and supportive environment. While CareBuddy can offer guidance and encouragement, it is not a substitute for professional therapy or medical advice. If you are in crisis or experiencing severe distress, please seek help from a licensed mental health professional or contact emergency services immediately.")
        
        st.subheader("Our mission:")
        st.write("Our mission is to empower individuals by providing accessible, empathetic, and practical mental health support. We aim to bridge the gap between daily stressors and professional help by offering thoughtful, evidence-based resources that foster resilience and well-being.")
        
        st.subheader("What we do:")
        st.write("CareBuddy is an intent-based chatbot designed to understand your unique mental health concerns and provide meaningful responses. Whether you’re seeking help with managing stress, coping with anxiety, navigating relationships, or improving self-care, we’re here to guide you.")
        
        st.subheader("Note:")
        st.write("While CareBuddy is here to support you, it is not a substitute for professional therapy or medical intervention. If you’re experiencing severe mental health challenges or are in crisis, please reach out to a licensed mental health professional or emergency services immediately.")
if __name__ == "__main__":
    main()