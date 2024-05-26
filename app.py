import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json
import os
import pandas as pd
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
client = MistralClient(api_key=api_key)

st.set_page_config(
    page_title="Psychological Safety Dashboard",
    layout="wide"
)

# Set page title
st.title("Conversate: Psychological Safety in the Workplace")

uploaded_file = st.file_uploader("Upload the slack conversation", type=['json'])

# Function to process uploaded file
@st.cache_data(show_spinner=False)
def process_file(uploaded_file):
    json_data = json.load(uploaded_file)
    messages_with_profile = [item for item in json_data if "user_profile" in item]

    conversation = []
    for item in messages_with_profile:
        d = {"user": item["user_profile"]["real_name"], "message": item["text"]}
        conversation.append(d)

    df = []
    for con in conversation:
        user = con['user']
        message = con['message']
        messages = [
            ChatMessage(
                role="user",
                content=f""" Given the message {message}, select the best label for the following categories that fits the message, omit any explanations or details and just respond in form of a python dictionary without any additional spaces or new lines. The dictionary should be in a single line without additional spaces. The keys and values of the dictionary must be enclosed with double quotes:\
                    sentiment = ['positive', 'negative', 'neutral'],\
                    emotions = [ 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'],\
                    toxicity = ['yes', 'no'],\
                    harassment indicators = ['harassment', 'hate speech', 'bullying', 'none'],\
                    sexist = ['yes', 'no'],\
                    spam = ['yes', 'no'],\
                    racism = ['yes', 'no'],\
                    profanity = ['yes', 'no']"""
            )
        ]

        # No streaming
        chat_response = client.chat(
            model=model,
            messages=messages,
        )

        res = json.loads(chat_response.choices[0].message.content)

        df.append({
            'user': user,
            'message': message,
            'sentiment': res['sentiment'],
            'emotions': res['emotions'],
            'toxicity': res['toxicity'],
            'harassment indicators': res['harassment indicators'],
            'sexist': res['sexist'],
            'spam': res['spam'],
            'racism': res['racism'],
            'profanity': res['profanity']
        })

    return pd.DataFrame(df)

# Load data if file is uploaded
if uploaded_file is not None:
    with st.spinner(f"Analyzing the conversation... (This may take a while depending on the size of the conversation)"):
        data = process_file(uploaded_file)

    default_options = list(set(data['user'])) + ['Entire Team']

    # Define color map for each emotion category
    color_map = {
        'admiration': ['#1f77b4', '#98df8a', '#2ca02c', '#d62728'],
        'amusement': ['#ff7f0e', '#98df8a', '#2ca02c', '#d62728'],
        'anger': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'annoyance': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'approval': ['#1f77b4', '#98df8a', '#2ca02c', '#d62728'],
        'caring': ['#98df8a', '#2ca02c', '#FF69B4', '#d62728'],
        'confusion': ['#ffbb78', '#ff7f0e', '#9467bd', '#d62728'],
        'curiosity': ['#ffbb78', '#ff7f0e', '#9467bd', '#d62728'],
        'desire': ['#2ca02c', '#ff7f0e', '#98df8a', '#d62728'],
        'disappointment': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'disapproval': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'disgust': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'embarrassment': ['#ffbb78', '#ff7f0e', '#9467bd', '#d62728'],
        'excitement': ['#ff7f0e', '#2ca02c', '#98df8a', '#d62728'],
        'fear': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'gratitude': ['#98df8a', '#2ca02c', '#1f77b4', '#d62728'],
        'grief': ['#ffbb78', '#d62728', '#bcbd22', '#ff7f0e'],
        'joy': ['#ff7f0e', '#98df8a', '#2ca02c', '#d62728'],
        'love': ['#FF69B4', '#98df8a', '#2ca02c', '#d62728'],
        'nervousness': ['#ffbb78', '#ff7f0e', '#9467bd', '#d62728'],
        'optimism': ['#98df8a', '#2ca02c', '#1f77b4', '#d62728'],
        'pride': ['#98df8a', '#ff7f0e', '#1f77b4', '#d62728'],
        'realization': ['#9467bd', '#ff7f0e', '#ffbb78', '#d62728'],
        'relief': ['#1f77b4', '#98df8a', '#2ca02c', '#d62728'],
        'remorse': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'sadness': ['#ffbb78', '#ff7f0e', '#d62728', '#bcbd22'],
        'surprise': ['#ff7f0e', '#ffbb78', '#9467bd', '#d62728'],
        'neutral': ['#2ca02c', '#98df8a', '#1f77b4', '#d62728']
    }

    with st.sidebar:

        # Create dropdown with default options
        selected_option = st.selectbox("Select the team member:", default_options)

        # Add submit button
        submit = st.button("Submit")

    if submit:
        with st.spinner('Processing messages by {s}...'.format(s=selected_option)):
            

            if selected_option == 'Entire Team':
                msgs = list(data['message'])
                messages = [
                ChatMessage(role="user", content=" Given the messages by the team, describe their communication style. What is they doing right and wrong and how should it be improved for effective communication. What must be done by this team to ensure pyschological safety in the team. Explain in detail. The messages by this team are {m}".format(m=msgs)
                            )
                ]
            else:
                msgs = list(data[data['user'] == selected_option]['message'])
                messages = [
                ChatMessage(role="user", content=" Given the messages by {s}, describe his/her communication style. What is he/she doing right and wrong and how should it be improved for effective communication. What must be done by this person to ensure pyschological safety in the team. Explain in detail. The messages by this person are {m}".format(s=selected_option, m=msgs)
                            )
                ]

            # No streaming
            chat_response = client.chat(
                model=model,
                messages=messages,
            ) 

            st.subheader(f"Team Member Selected: {selected_option}")

            with st.expander("See explanation"):
                st.write(chat_response.choices[0].message.content)


            st.write("")
            st.write("")
            st.write("")

            senti, _, emot = st.columns([4.5,1,4.5])

            with senti:
            
                if selected_option == 'Entire Team':
                    sentiment_labels = list(dict(data['sentiment'].value_counts()).keys())
                    sentiment_values = list(dict(data['sentiment'].value_counts()).values())
                else:
                # Sentiment counts
                    sentiment_labels = list(dict(data[data['user'] == selected_option]['sentiment'].value_counts()).keys())
                    sentiment_values = list(dict(data[data['user'] == selected_option]['sentiment'].value_counts()).values())

                # Define colors for each sentiment category
                colors = ['lightblue', 'lightcoral', 'lightgreen']

                # Create a pie chart
                fig = go.Figure(data=[go.Pie(labels=sentiment_labels, values=sentiment_values)])

                # Update pie chart layout
                fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                                marker=dict(colors=colors, line=dict(color='#000000', width=2)))

                # Set title
                # fig.update_layout(title='Sentiment Distribution')
                st.subheader("Sentiment Distribution")

                # Display pie chart
                st.plotly_chart(fig, use_container_width=True)

            with _:
                st.write("")

            with emot:

                if selected_option == 'Entire Team':
                    emotion_labels = list(dict(data['emotions'].value_counts()).keys())
                    emotion_values = list(dict(data['emotions'].value_counts()).values())
                else:
                
                    emotion_labels = list(dict(data[data['user'] == selected_option]['emotions'].value_counts()).keys())
                    emotion_values = list(dict(data[data['user'] == selected_option]['emotions'].value_counts()).values())

                predicted_probabilities_ED = [count / sum(emotion_values) for count in emotion_values]

                top_emotions = emotion_labels[:4]
                top_scores = predicted_probabilities_ED[:4]
                # Create the gauge charts for the top 4 emotion categories
                fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                                                            [{'type': 'indicator'}, {'type': 'indicator'}]],
                                    vertical_spacing=0.4)

                for i, emotion in enumerate(top_emotions):
                    # Get the emotion category, color, and normalized score for the current emotion
                    category = emotion
                    color = color_map[category]
                    value = top_scores[i] * 100
                    
                    # Calculate the row and column position for adding the trace to the subplots
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    # Add a gauge chart trace for the current emotion category
                    fig.add_trace(go.Indicator(
                        domain={'x': [0, 1], 'y': [0, 1]},
                        value=value,
                        mode="gauge+number",
                        title={'text': category.capitalize()},
                        gauge={'axis': {'range': [None, 100]},
                            'bar': {'color': color[3]},
                            'bgcolor': 'white',
                            'borderwidth': 2,
                            'bordercolor': color[1],
                            'steps': [{'range': [0, 33], 'color': color[0]},
                                        {'range': [33, 66], 'color': color[1]},
                                        {'range': [66, 100], 'color': color[2]}],
                            'threshold': {'line': {'color': "black", 'width': 4},
                                            'thickness': 0.5,
                                            'value': 50}}), row=row, col=col)

                # Update the layout of the figure
                fig.update_layout(height=400, margin=dict(t=50, b=5, l=0, r=0))


                # Display gauge charts
                
                st.subheader("Emotion Detection")
            
                st.plotly_chart(fig, use_container_width=True)

            st.write("")
            st.write("")
            st.write("")

            tox, rac, sex, spam, prof = st.columns([2,2,2,2,2])

            with tox:

                #Toxicity
                toxicity = False
                if selected_option == 'Entire Team':
                    toxicity = 'yes' in dict(data['toxicity'].value_counts())
                else:
                    toxicity = 'yes' in dict(data[data['user'] == selected_option]['toxicity'].value_counts())

                if toxicity:
                    st.subheader("Toxicity detected in the conversation.")
                    st.image(f"imgs/toxic_yes.jpeg", width=200)
                else:
                    st.subheader("No toxicity detected in the conversation.")
                    st.image(f"imgs/toxic_no.jpeg", width=200)

            with rac:
                #Racism
                racism = False
                if selected_option == 'Entire Team':
                    racism = 'yes' in dict(data['racism'].value_counts())
                else:
                    racism = 'yes' in dict(data[data['user'] == selected_option]['racism'].value_counts())

                if racism:
                    st.subheader("Racism detected in the conversation.")
                    st.image(f"imgs/racism_yes.jpeg", width=200)
                else:
                    st.subheader("No racism detected in the conversation.")
                    st.image(f"imgs/racism_no.jpeg", width=200)

            with sex:
                #Sexism
                sexism = False
                if selected_option == 'Entire Team':
                    sexism = 'yes' in dict(data['sexist'].value_counts())
                else:
                    sexism = 'yes' in dict(data[data['user'] == selected_option]['sexist'].value_counts())

                if sexism:
                    st.subheader("Sexism detected in the conversation.")
                    st.image(f"imgs/sexism_yes.png", width=200)
                else:
                    st.subheader("No sexism detected in the conversation.")
                    st.image(f"imgs/sexism_no.jpeg", width=200)

            with spam:
                #spam
                spam = False
                if selected_option == 'Entire Team':
                    spam = 'yes' in dict(data['spam'].value_counts())
                else:
                    spam = 'yes' in dict(data[data['user'] == selected_option]['spam'].value_counts())

                if spam:
                    st.subheader("Spam detected in the conversation.")
                    st.image(f"imgs/spam_yes.jpeg", width=200)
                else:
                    st.subheader("No spam detected in the conversation.")
                    st.image(f"imgs/spam_no.png", width=200)

            with prof:
                #profanity
                profanity = False
                if selected_option == 'Entire Team':
                    profanity = 'yes' in dict(data['profanity'].value_counts())
                else:
                    profanity = 'yes' in dict(data[data['user'] == selected_option]['profanity'].value_counts())

                if profanity:
                    st.subheader("Profanity detected in the conversation.")
                    st.image(f"imgs/profanity_yes.jpeg", width=200)
                else:
                    st.subheader("No profanity detected in the conversation.")
                    st.image(f"imgs/profanity_no.jpeg", width=200)

            st.write("")
            st.write("")
            st.write("")

            _, har, __ = st.columns([4,2,4])

            with _:
                st.write("")

            with har:
            
                # Harassment Indicator
                if selected_option == 'Entire Team':
                    harassment = list(dict(data['harassment indicators'].value_counts()).keys())
                else:
                    harassment = list(dict(data[data['user'] == selected_option]['harassment indicators'].value_counts()).keys())
                
                filtered_values = [value for value in harassment if value != 'none']

                if len(filtered_values) > 0:
                    st.subheader(f"Harassment indicators detected in the conversation: {', '.join(filtered_values)}",)
                    st.image(f"imgs/harass_yes.jpeg", width=200)
                else:
                    st.subheader("No harassment indicators detected in the conversation.")
                    st.image(f"imgs/harass_no.jpeg", width=200)

            with __:
                st.write("")


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)