import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import io
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2, requests, os
import numpy as np
import json


# url = "http://api:5000/save/"

def detect_orientation(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Compute the angle of the first detected line
    if lines is not None and len(lines) > 0:
        x1, y1, x2, y2 = lines[0][0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle
    
    return None

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Get the center of the image
    center = (width // 2, height // 2)
    
    # Perform rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated_image

def get_pdf_text(pdf_file):
    try:
        # Read PDF file as bytes
        pdf_bytes = pdf_file.read()

        # Convert PDF to image(s)
        images = convert_from_bytes(pdf_bytes)

        # Extract text from each image and combine
        text = ""
        for image in images:
            # Convert the image to a NumPy array
            image_np = np.array(image)

            # Detect orientation of the image
            angle = detect_orientation(image_np)
            
            # If angle is not None, rotate the image
            if angle is not None:
                image_np = rotate_image(image_np, angle)
            
            # Convert the NumPy array back to an image
            rotated_image = Image.fromarray(image_np)

            # Extract text from the rotated image
            image_text = pytesseract.image_to_string(rotated_image)
            text += image_text + "\n"  # Add a new line between each image's text
        
        return text
    except Exception as e:
        st.error("Failed to extract text from PDF using Tesseract OCR: {}".format(e))
        return ""

def clean_data(text):
    # Remove unnecessary characters and split the text into lines
    cleaned_text = [line.strip() for line in text.split('\n') if line.strip()]

    # Join the cleaned lines back into a single string
    cleaned_text = '\n'.join(cleaned_text)

    # Print or use the cleaned text
    return cleaned_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="distilbert-base-nli-mean-tokens")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=.7)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"max_new_tokens": 100})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, pdf_id):
    bot_response = ""  # Variable to store the bot's response

    if pdf_id in st.session_state.conversations:
        conversation = st.session_state.conversations[pdf_id]
        if conversation is not None:
            response = conversation({'question': user_question})
            st.session_state.conversations[pdf_id] = response['chat_history']

            for i, message in enumerate(st.session_state.conversations[pdf_id]):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                    bot_response = message.content  # Store the bot's response
        else:
            st.error("Conversation chain not initialized for PDF file '{}'.".format(pdf_id))
    else:
        st.error("No conversation chain found for PDF file '{}'.".format(pdf_id))

    return bot_response

def main():
    load_dotenv()
    st.set_page_config(page_title="Extract Information from PDFs")
    st.write(css, unsafe_allow_html=True)

    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                for pdf_file in pdf_docs:
                    # Clear previous conversation chain for this PDF
                    pdf_id = pdf_file.name  # Using file name as ID
                    st.session_state.conversations[pdf_id] = None

                    # get pdf text
                    text = get_pdf_text(pdf_file)
                    print("text: ", text)

                    clean_text = clean_data(text)
                    print("clean text: ", clean_text)

                    # get the text chunks
                    text_chunks = get_text_chunks(clean_text)
                    print("chunks: ", text_chunks)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversations[pdf_id] = get_conversation_chain(
                        vectorstore)
                    
                    print("Conversation Chain for {}: {}".format(pdf_id, st.session_state.conversations[pdf_id]))

                    user_question = """
                    Extract all the following values: Certificate No, Invoice No and Steel Grade.

                    Response format: JSON
                    """
                    response = handle_userinput(user_question, pdf_id)

                    if response:
                        # try:
                            data = json.loads(response)
                            st.json(data)
                            
                            # Enviar a resposta JSON para a API Flask
                        #     post_response = requests.post(url, data={'pdf_id': pdf_id, 'data': json.dumps(data)})
                            
                        #     if post_response.status_code == 200:
                        #         st.success("Data successfully sent to the API.")
                        #     else:
                        #         st.error(f"Failed to send data to the API. Status code: {post_response.status_code}")
                        # except json.JSONDecodeError:
                        #     st.error("Failed to parse the bot response as JSON.")


if __name__ == '__main__':
    main()
