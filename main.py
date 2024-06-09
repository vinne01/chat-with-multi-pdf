import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

st.sidebar.image("https://ik.imagekit.io/vinaymry/WhatsApp%20Image%202024-06-07%20at%204.52.06%20PM%20(1).jpeg?updatedAt=1717759434849&tr=n-ik_ml_thumbnail", use_column_width=True)
# ICON_BLUE = "https://ik.imagekit.io/vinaymry/WhatsApp%20Image%202024-06-04%20at%2011.39.45%20PM%20(1).jpeg?updatedAt=1717762600149"
#
# options = [ ICON_BLUE]
# sidebar_logo = st.selectbox("Sidebar logo", options, 0)
# main_body_logo = st.selectbox("Main body logo", options, 1)

# st.logo(sidebar_logo, icon_image=main_body_logo)
# st.sidebar.markdown("Hi!")
load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


#####################################################
# FUNCTION TO READ THE TEXT FROM PDFs
#####################################################
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


#####################################################
# DIVIDE THE TEXT FROM PDFs INTO CHUNKS
#####################################################
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


#####################################################
# GET TEXT EMBEDDINGS USING GEMINI MODEL
#####################################################
def get_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(chunks, embedding=embeddings)
    vectors.save_local("faiss_index")


#####################################################
# PREPARING THE CONVERSATIONAL CHAIN
#####################################################
def get_conversational_chain():
    prompt_temp = '''
    Answer the question from the provided context. Try to answer in as detailed manner as possible from the provided context.
    If the answer to the question is not known from the provided context, then dont provide wrong answers, in that case just say,
    'Answer to the question is not available in the provided document. Feel free to ask question from the provided context.'
    Context:\n{context}?\n
    Question:\n{question}\n
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5)

    prompt = PromptTemplate(
        template=prompt_temp,
        input_variables=['context', 'question']
    )

    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain


#####################################################
# PREPARING THE MODEL'S RESPONSE
#####################################################
def get_response(user_input):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)
    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': user_input},
        return_only_outputs=True
    )
    print(response)
    st.write('Reply: ', response['output_text'])


#####################################################
# CREATING THE FRONT END APPLICATION
#####################################################
def main():
    st.title('Chat With Multiple PDF 🤖')
    st.subheader('Ask any question related to the uploaded PDFS')
    st.sidebar.page_link("https://myfirstsportfolio.netlify.app/", label="See my portfolio")
    user_question = st.text_input(
        label='Ask a question related to the uploaded PDFs.',
        label_visibility='hidden',
        placeholder='Type your question here...'
    )
    if user_question and len(user_question) > 0:
        get_response(user_question)
    with st.sidebar:
        st.title('Upload PDF 📄')
        documents = st.file_uploader(
            label='Upload PDFs',
            label_visibility='hidden',
            type=['pdf'],
            accept_multiple_files=True
        )
        if st.button('Submit and Process'):
            with st.spinner('In Process...'):
                text = get_pdf_text(documents)
                chunks = get_text_chunks(text)
                get_embeddings(chunks)
                st.success('DONE!')


if __name__ == '__main__':
    main()