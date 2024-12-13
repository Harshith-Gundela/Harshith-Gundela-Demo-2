import os
import base64
import pandas as pd
import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory  
from dotenv import load_dotenv
from functools import cache
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv()

# Environment Variables

os.environ['AZURE_OPENAI_API_KEY'] = "b2f65bbb2bc84ff4b7388916ab18e332"
os.environ['AZURE_OPENAI_ENDPOINT'] = "https://zionaz.openai.azure.com/"
os.environ['AZURE_OPENAI_DEPLOYMENT'] = "gpt-35-turbo-finetuned"
os.environ['AZURE_OPENAI_API_VERSION'] = "2023-05-15"

# Load Dataset
try:
    dset = pd.read_csv("Data.csv")
except FileNotFoundError:
    st.error("Data.csv file not found.")
    st.stop()

# Initialize Azure Chat OpenAI
llm = AzureChatOpenAI(
    openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT'],
    openai_api_version=os.environ['AZURE_OPENAI_API_VERSION']
)

st.set_page_config(page_title="ZionAi HCP Chat", page_icon="zionicon.ico", layout="wide")

# Memory Initialization
try:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
except Exception as e:
    st.error(f"Memory initialization failed: {e}")
    memory = None

# Create Agent
agent = create_pandas_dataframe_agent(llm, dset, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)


def chunk_text(text, max_token_length):
    tokens = text.split()
    for i in range(0, len(tokens), max_token_length):
        yield ' '.join(tokens[i:i + max_token_length])

# Summarize Long Responses
def summarize_long_response(response, threshold=2000):
    if len(response.split()) > threshold:
        summary_prompt = f"Summarize the following text concisely:\n\n{response}"
        summary = agent.invoke(summary_prompt)
        return summary.get('output', response) if isinstance(summary, dict) else summary
    return response

@cache
def ask_agent(question):
    st.session_state.conversation_history.append({"role": "user", "content": question})
    try:
        max_input_tokens = 3000
        question_chunks = list(chunk_text(question, max_input_tokens))
        all_responses = []

        for chunk in question_chunks:
            response = agent.invoke(chunk)
            output = response.get('output') if isinstance(response, dict) else response
            all_responses.append(output)

        full_response = " ".join(all_responses)
        summarized_response = summarize_long_response(full_response)

        st.session_state.conversation_history.append({"role": "assistant", "content": summarized_response})

        return f"{summarized_response}"
        
    except ValueError as e:
        st.error(f"Parsing Error: {e}")
        return "The agent encountered a parsing error. Please try rephrasing your query."

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Image file {image_path} not found.")
        return None
image_base64 = get_base64_image("ZCS.png")

# image_path = "zionicon.png"  
# st.image(image_path, caption="", use_column_width=False, width=100)
st.markdown(
    f"""
    <style>
     /* Animated background */
        body {{
            background: linear-gradient(120deg, rgba(61,91,169,0.8), rgba(166,212,154,0.8));
            animation: backgroundAnimation 10s ease-in-out infinite;
            background-size: 400% 400%;
            color: #333;
        }}
        
        @keyframes backgroundAnimation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .stApp {{
            background-image: url("data:image/png;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #333;
        }}
        .sub-title, .section-title, .stTextInput label, h1, h2, h3, h4, h5, h6, p {{
            font-weight: bold;
            background: linear-gradient(to right, #3D5BA9, #A6D49A);
            -webkit-background-clip: text;
            color: transparent;
        }}
        .subheading-title span {{
            font-size: 2 rem;
            flex: 1;
            background: linear-gradient(to right, #3D5BA9, #A6D49A);
            -webkit-background-clip: text;
            color: transparent;
        }}
         .main-title span {{
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin: 0;
            flex: 1;
            background: linear-gradient(to right, #3D5BA9, #A6D49A);
            -webkit-background-clip: text;
            color: transparent;
        }}
        .stTextInput > div > div > input {{
            background-color: #f0f8ff;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #3D5BA9;
        }}
        .stButton > button {{
            background-color: #3D5BA9;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 24px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }}
        .stButton > button:hover {{
            background-color: #A6D49A;
            color: #333;
        }}
         /* Enhanced Navigation styling */
        .nav {{
            display: flex;
            gap: 1.5rem;
            font-size: 1.1rem;
        }}
        
        .nav a {{
            color: #3D5BA9;
            text-decoration: none;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 12px;
            transition: all 0.3s ease;
            position: relative;
        }}
        
        .nav a:hover {{
            color: #ffffff;
            background-color: #A6D49A;
            box-shadow: 0px 4px 8px rgba(61, 91, 169, 0.2);
            transform: translateY(-2px);
        }}
        
        .nav a::before {{
            content: '';
            position: absolute;
            top: -3px;
            left: -3px;
            right: -3px;
            bottom: -3px;
            border-radius: 15px;
            background: linear-gradient(45deg, rgba(61,91,169,0.5), rgba(166,212,154,0.5));
            z-index: -1;
            transition: opacity 0.3s ease;
            opacity: 0;
        }}
        
        .nav a:hover::before {{
            opacity: 1;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
# st.markdown(
#     f"""
#     <div class="header">
#         <nav class="nav">
#             <a href="https://zionai.com/about-us/">ZionAI</a>
#             <a href="https://zionclouds.com/">Zion Cloud Solutions</a>
#             <a href="https://www.linkedin.com/in/arun-reddy-9368863/">Arun Reddy</a>
#             <a href="https://www.linkedin.com/in/sriya-samhitha-madduri-1325a6230/">Sriya Samhitha</a>
#         </nav>
#     </div>
    
#     """,
#     unsafe_allow_html=True
# )

def embed_powerbi_report(report_url):
    st.markdown(
        f'<iframe width="100%" height="600" src="{report_url}" frameborder="0" allowFullScreen="true"></iframe>',
        unsafe_allow_html=True
    )
image_path = "zionai.png"


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


image_base64 = get_base64_image(image_path)


st.markdown(
    f"""
    <style>
    .bottom-right-image {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 100px;
    }}
    </style>
    <img src="data:image/png;base64,{image_base64}" class="bottom-right-image">
    """,
    unsafe_allow_html=True
)

image_path = "zionicon.png"  
st.image(image_path, caption="", use_column_width=False, width=100)
st.markdown("<h1 class='main-title'>Health Care Practitioner Engagement</h1>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Data Insights", "Power BI Visualization", "Chat History"])
st.markdown(
    """
     <style>
    /* Sidebar background with glassmorphism effect */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }

    /* Sidebar title with soft neon glow */
    [data-testid="stSidebar"] h1 {
        font-weight: bold;
        background: linear-gradient(to right, #3D5BA9, #A6D49A);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        font-family: 'Roboto', sans-serif;
        margin-top: 0;
        margin-bottom: 25px;
        text-shadow: 0px 0px 8px rgba(255, 255, 255, 0.6);
    }

    /* Page selection options styled as glowing buttons */
    .stSelectbox > div {
        font-size: 16px;
        color: #e0e0e0;
        font-weight: 500;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        font-family: 'Roboto', sans-serif;
        transition: background-color 0.3s ease, transform 0.2s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }

    /* Hover effect with gradient glow */
    .stSelectbox > div:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(0, 150, 255, 0.3));
        box-shadow: 0px 4px 15px rgba(0, 150, 255, 0.4);
        transform: translateY(-3px);
    }

    /* Selected option glow effect */
    .stSelectbox > div:focus {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(0, 150, 255, 0.5);
        box-shadow: 0px 4px 15px rgba(0, 150, 255, 0.4);
    }

    /* Subtle animation effect for the sidebar on hover */
    [data-testid="stSidebar"]:hover {
        transform: scale(1.02);
    }

    /* Smooth scroll effect with custom glow scrollbar */
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6c63ff, #3a3d99);
        border-radius: 8px;
    }

    /* Fonts and padding reset */
    .css-18e3th9 {
        padding: 0;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Styling for the text input */
    div[data-baseweb="input"] > div {
        border: 2px solid #3D5BA9 !important;
        border-radius: 5px;
        transition: border-color 0.3s ease-in-out;
    }
    /* Hover effect for the text input */
    div[data-baseweb="input"] > div:hover {
        border-color: #A6D49A !important;
        box-shadow: 0px 0px 8px #A6D49A !important;
    }
    /* Adjusting the input field itself */
    div[data-baseweb="input"] > div > input {
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def embed_filtered_powerbi_report(report_url, filter_column, filter_value):
    
    filtered_url = f"{report_url}&$filter={filter_column} eq '{filter_value}'"
    st.markdown(
        f'<iframe width="100%" height="600" src="{filtered_url}" frameborder="0" allowFullScreen="true"></iframe>',
        unsafe_allow_html=True
    )
icon_path = "zionicon.png"

def display_response_with_icon(response_text):
    col1, col2 = st.columns([0.1, 0.9])  
    with col1:
        st.image(icon_path, width=30)  
    with col2:
        st.write(response_text)
if page == "Home":
    user_question = st.text_input("Enter your query")

    if user_question:
        response = ask_agent(user_question)
        st.subheader("Response:")
        # st.write(response)
        display_response_with_icon(response)

elif page == "Data Insights":
    st.markdown("<h2 class='subheading-title'>Data Insights</h2>", unsafe_allow_html=True)
    user_question = st.text_input("Ask a question for insights:")
    
    if user_question:
        filter_column = "DesiredColumn"  
        embed_filtered_powerbi_report(
            "https://app.powerbi.com/reportEmbed?reportId=8962640d-3da2-4b6c-8f9a-485578d2389b&autoAuth=true&ctid=e28d23e3-803d-418d-a720-c0bed39f77b6",
            filter_column, 
            user_question
        )

elif page == "Power BI Visualization":
    st.markdown("<h2 class='subheading-title'>Power BI Report</h2>", unsafe_allow_html=True)
    report_url = "https://app.powerbi.com/reportEmbed?reportId=8962640d-3da2-4b6c-8f9a-485578d2389b&autoAuth=true&ctid=e28d23e3-803d-418d-a720-c0bed39f77b6"
    embed_powerbi_report(report_url)

elif page == "Chat History":
    st.markdown("<h2 class='subheading-title'>Chat History</h2>", unsafe_allow_html=True)
    if st.session_state.conversation_history:
        for entry in st.session_state.conversation_history:
            st.write(f"**{entry['role'].capitalize()}**: {entry['content']}")
    else:
        st.write("No chat history available.")


