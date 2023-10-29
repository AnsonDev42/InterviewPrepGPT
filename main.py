import dotenv
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

''' Load files '''
dotenv.load_dotenv()
loader = TextLoader("supporting_docs/Yaowen_Shen_SWE_v1.1.txt")
resume = loader.load()

loader = TextLoader("supporting_docs/wise_jd.txt")
jd = loader.load()

st.title('🦜🔗 Interview Preparation App')
from langchain.callbacks.base import BaseCallbackHandler

openai_api_key = dotenv.get_key('.env', 'OPENAI_API_KEY')


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + "/"
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


chat_box = st.empty()
stream_handler = StreamHandler(chat_box, display_method='write')


template = """You are an AI chatbot having a conversation with a human, to help them with their interview as new graduate software engineer; You know the user based on {resume} and {history} 
{resume}
Here is the job description:
{jd}
{chat_history}

Human: {human_input}
AI: """

prompt = PromptTemplate(input_variables=["resume", "chat_history", "human_input","jd"], template=template)
memory = ConversationBufferMemory(memory_key="history", input_key='human_input', )
ConversationBufferMemory.
# streamlit generate a text window to upload job description

jd = ""
job_description_raw = st.file_uploader('Upload a job description', type=['txt'])
jd_placeholder = st.empty()
if job_description_raw is not None:
    jd = jd_placeholder.text_area("Press cmd+enter to update any change",job_description_raw.getvalue().decode("utf-8"),label_visibility='visible')
    print('!!!!!!!!!!!!!!!!!!!!!!!!')
    print(jd)
# streamlit generate a text window to upload resume

# Generate LLM response
def generate_response(csv_file, input_query):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2, openai_api_key=openai_api_key)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    message_placeholder = st.empty()
    response = llm_chain({"chat_history": "","jd": jd, "resume": resume, "human_input": input_query}, return_only_outputs=True)
    # add response into the memory
    memory
    message_placeholder.markdown(response['text'])
    return st.success("How's that? try ask me more questions!")


# Input widgets
# uploaded_file = st.file_uploader('Upload a CSV/pdf file', type=['csv', 'pdf'])
uploaded_file= 'WIP'
question_list = [
    'Where did I graduate from?', # test question
    "What's my contribution in my project?",
    'Other']

query_text = st.text_area('Ask GPT4 to prepare you interview question:', 'Your understanding of the expectations of this particular role, and why we are hiring this person to help us. Where do you see your impact?', )  # disabled=not uploaded_file
button = st.button('Submit')

# App logic
if query_text == 'Other':
    query_text = st.text_input('Enter your query:', placeholder='Enter query here ...', disabled=uploaded_file)
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
if button:
    if openai_api_key.startswith('sk-') and (uploaded_file is not None):
        st.header('Output')
        generate_response(uploaded_file, query_text)
    if openai_api_key.startswith('sk-') and (uploaded_file is None):
        st.header('No file Output')
        generate_response(uploaded_file, query_text)
