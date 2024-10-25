import streamlit as st
st.set_page_config(layout="wide")
import boto3
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from botocore.config import Config
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Streamlit configuration
st.title("Architectural Blueprints Analysis")
st.write("Enter the S3 path to an Architectural Blueprints PDF document for data extraction and analysis.")

# AWS Configuration
retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={'max_attempts': 10, 'mode': 'standard'}
)
session = boto3.Session()
bedrock = session.client(
    service_name='bedrock-runtime', 
    aws_access_key_id=os.environ.get("aws_access_key_id"),
    aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    config=retry_config
)

# Function to initialize model
def get_model():
    model = ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL"),
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
    )
    return model 

# Function to load data from S3
def load_data(s3_path):
    loader = AmazonTextractPDFLoader(s3_path)
    docs = loader.load()
    return docs

# Function to get LLM response
def get_llm_response(model, data, output_parser):
    prompt = PromptTemplate(
        template="""
            You are an AI assistant specializing in extracting structured data from complex Architectural Blueprints. Your task is to analyze the provided TEXT data, 
            which contains text, tables, and forms extracted from a Architectural Blueprintss PDF document. It is CRITICAL that you extract information from it, like what are the blueprints about, distances, areas, locations, and any possible information.
            IMPORTANT INSTRUCTIONS:
            1. Analyze EVERY SINGLE ELEMENT of the provided data: all text, every cell in every table, and all form fields.
            2. If the table is detected return it in proper tabular format.
            3. Process ALL pages and ALL data in the document.           
            Here is the TEXT data extracted from the Architectural Blueprints pdf:
            {data}""",
        input_variables=["data"]
    )
    
    chain = prompt | model | output_parser
    response = chain.invoke({"data": data})
    return response

# Streamlit App UI
s3_path = st.text_input("Enter S3 Path", "")

if st.button("Extract Data"):
    if s3_path:
        st.write("Loading data from S3...")
        data = load_data(s3_path)
        model = get_model()
        output_parser = StrOutputParser()
        
        st.write("Processing data with LLM...")
        extracted_info = get_llm_response(model=model, data=data, output_parser=output_parser)
        
        st.write("Extracted Information:")
        st.text(extracted_info)
    else:
        st.warning("Please enter a valid S3 path.")
