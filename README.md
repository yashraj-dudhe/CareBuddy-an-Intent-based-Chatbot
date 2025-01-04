# CareBuddy - Mental Health Chatbot 🤖💙

CareBuddy is a mental health chatbot designed to provide accessible support, guidance, and resources for emotional well-being. Using advanced Natural Language Processing (NLP) techniques and machine learning models, CareBuddy can classify user intents, answer frequently asked questions (FAQs), and offer empathetic responses to help manage stress, anxiety, and other mental health challenges.

---

## 🚀 Features

- **Intent Recognition**: Classifies user input into predefined categories using Logistic Regression and TF-IDF vectorization.
- **FAQ Matching**: Employs Sentence Transformers and cosine similarity to provide context-aware responses to common questions.
- **Interactive Interface**: Built using Streamlit for a seamless and user-friendly experience.
- **Fallback Responses**: Handles out-of-scope queries with generic but supportive responses.
- **Conversation Logging**: Logs interactions for monitoring and improvement.
- **Mental Health Awareness**: Promotes self-care and emotional well-being.

---

## 🛠️ Tools and Technologies

1. **Programming Language**: Python
2. **NLP Libraries**:  
   - NLTK  
   - Sentence Transformers  
   - Scikit-learn (TF-IDF Vectorizer and Logistic Regression)
3. **Frontend Framework**: Streamlit
4. **Machine Learning**: Logistic Regression, BERT-based embeddings
5. **Utilities**:  
   - `csv` for logging conversations  
   - `datetime` for timestamps  
   - `os` and `json` for file handling

---

## 🧠 Methodology

1. **Data Preparation**  
   - Intent data stored in a structured JSON format.
   - FAQ dataset created for mental health-related queries.

2. **Model Development**  
   - TF-IDF vectorizer converts user inputs into numerical features.
   - Logistic Regression model classifies inputs into predefined intents.
   - Sentence Transformers generate embeddings for FAQ questions and user queries.

3. **Response Mechanism**  
   - Classified intents or matched FAQs return specific responses.
   - Fallback mechanism ensures meaningful engagement for unrecognized queries.

4. **Interface and Deployment**  
   - Streamlit provides an interactive web-based interface for user interactions.
   - Conversation history logged for analysis and improvement.

---
### Screenshots
![Carebuddy](/img.png)

### 🤖Try now:
- **CareBuddy**: [Link](https://carebuddy.streamlit.app/)
  
### Prerequisites
- Python 3.8 or above
- Required Python libraries: `nltk`, `streamlit`, `scikit-learn`, `sentence-transformers`


### Description of Key Files

1. **nlp.py**  
   Contains the core logic for the chatbot, including intent classification, FAQ matching, response generation, and Streamlit interface.

2. **intents.json**  
   A structured dataset defining user intents, patterns, and corresponding responses.

3. **requirements.txt**  
   A list of dependencies and libraries required to run the chatbot.

4. **chat_log.csv**  
   Logs user inputs, chatbot responses, and timestamps during interactions for analysis and improvement. This file is created dynamically during runtime.

5. **Week 3 Project PPT**  
   A PowerPoint presentation detailing the project's methodology, tools, and findings.

6. **assets/**  
   A directory for storing additional resources like pre-downloaded datasets or NLTK data files.

---

## 📬 Contact

If you have any questions, feedback, or suggestions, feel free to reach out:

- **Author**: Yashraj Dudhe  
- **Email**: [yashraj.dudhe110@gmail.com](mailto:yashraj.dudhe110@gmail.com)  
- **GitHub**: [yasraj-dudhe](https://github.com/yashraj-dudhe)  
- **LinkedIn**: [Yashraj Dudhe](https://www.linkedin.com/in/yashrajdudhe)  

We appreciate your interest in CareBuddy and look forward to hearing from you!


