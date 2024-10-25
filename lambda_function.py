import boto3
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from botocore.config import Config
import os
# import pytesseract
# from pdf2image import convert_from_path
from langchain_core.prompts import PromptTemplate

retry_config = Config(
    region_name=os.environ.get("region_name"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id = os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key = os.environ.get("aws_secret_access_key"),
                                       config=retry_config)

def get_model():
    model = ChatBedrock(
        model_id = os.getenv("BEDROCK_MODEL"),
        client = bedrock,
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.9,
        },
    )
    return model 

def load_data(s3_path):
    loader = AmazonTextractPDFLoader(s3_path)
    docs = loader.load()
    return docs

def get_llm_response(model, 
                     data,
                     output_parser):
    prompt = PromptTemplate(
        template="""
                You are an AI assistant specializing in extracting structured data from complex engineering drawing. Your task is to analyze the provided TEXT data, 
                which contains text, tables, and forms extracted from a engineering drawings PDF document. It is CRITICAL that you extract information from it, like what are the blueprints about, distances, areas, locations, and any possible information.
                IMPORTANT INSTRUCTIONS:
                1. Analyze EVERY SINGLE ELEMENT of the provided data: all text, every cell in every table, and all form fields.
                2. If the table is detected extract it in proper tabular format.
                3. Process ALL pages and ALL data in the document.
                
                Here is the TEXT data extracted from the engineering drawing pdf:
                {data}""",
        input_variables=["data"],
        )
    
    chain = prompt | model | output_parser
    
    resp = chain.invoke({input:data})
    return resp

# def extract_text_from_pdf(pdf_path):
#     # Convert PDF to images (one image per page)
#     images = convert_from_path(pdf_path)
#     extracted_text = ""

#     # Perform OCR on each page image
#     for page_num, img in enumerate(images):
#         print(f"Processing page {page_num + 1}")
#         text = pytesseract.image_to_string(img)
#         extracted_text += text + "\n"

#     return extracted_text


# pdf_path = "/home/chandan/Projects/engg_drawings/sample_files/04-501.pdf"
# text = extract_text_from_pdf(pdf_path=pdf_path)

def lambda_handler(event, context):
    file_path = event.get("file_path")
    data = load_data(file_path)
    model = get_model()
    output_parser = StrOutputParser()
    
    extracted_info = get_llm_response(model = model,
                                data=data,
                                output_parser = output_parser)
    
    return {
        'statusCode': 200,
        'body': extracted_info
    }