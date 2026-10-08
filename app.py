
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel,Field,field_validator
import streamlit as st

load_dotenv()

os.environ['GOOGLE_API_KEY']=st.secrets('gemini_key')

#Prompt template
prompt_template=ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template='Respond to user queries concisely'),
    HumanMessagePromptTemplate.from_template(template='Explian {topic} with {number} examples.')
])

#llm
llm=init_chat_model('google_genai:gemini-3.8-flash')

#parser
parser=StrOutputParser()

chain=prompt_template | llm | parser

class InputValidation(BaseModel):
    topic:str = Field(default='machine learning',description='Any topic name that user want to know about',min_length=5)
    number:int=Field(default=3,description='Number of examples user want llm to use for explanation',gt=0)

    @field_validator('topic')
    @classmethod
    def clean_topic(cls,topic):
        return topic.strip()

class ResponseValidation(BaseModel):
    result:str = Field(description='This is the explanation returned by the llm')

    @field_validator('result')
    @classmethod
    def clean_result(cls,result):
        return result.strip()


def response(topic:str,number:int)->str:
    input=InputValidation(topic=topic,number=number)
    result=chain.invoke({'topic':input.topic,'number':input.number})
    output = ResponseValidation(result=result)
    return output.result
    
# print(response(topic='deep learning',number=2))

#Title
st.set_page_config(
    page_title="Topic Explainer",
    page_icon="📚"
)
st.title("📚 Topic Explianer")

#Text box for topic and numbers
topic = st.text_input("Enter topic")
number = st.number_input("Enter number of examples",min_value=1,value=3,step=1)

#Button
if st.button("Explain"):

    if not topic.strip():

        st.error("Please enter a topic.")
    else:

        with st.spinner("Generating response..."):

            try:

                result = response(
                    topic= topic.strip(),
                    number= int(number)
                )

                st.subheader("Response")
                st.write(result)
            

            except Exception as e:
                st.error(f"Error: {str(e)}")

