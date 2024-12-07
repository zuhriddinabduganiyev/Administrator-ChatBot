import streamlit as st

# Update these imports
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
# If using OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class NajotTalimBot:
    def __init__(self):
        # Initialize application
        self.setup_page_config()
        self.add_custom_css()
        self.initialize_session_state()

    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Najot Ta'lim Chatbot",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

    def add_custom_css(self):
        """Custom CSS for styling"""
        st.markdown("""
        <style>
        body { background-color: #343541; color: #d1d5db; }
        [data-testid="stSidebar"] { background-color: #d1d5db; }
        .stTextInput > div > div > input { color: #fff; background-color: #444654; }
        .stButton > button { background-color: #10a37f; color: white; }
        .stButton > button:hover { background-color: #15be8f; }
        </style>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'thread_id' not in st.session_state:
            st.session_state['thread_id'] = str(uuid.uuid4())
        if 'openai_api_key' not in st.session_state:
            st.session_state['openai_api_key'] = None

    def load_pdf_and_initialize_vectorstore(self):
        """Load PDF data and set up vector store for embedding-based retrieval"""
        try:
            # Load PDF file
            pdf_loader = PyPDFLoader('najot_talim.pdf')  # Replace with your actual file path
            documents = pdf_loader.load()

            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)

            # Create embeddings and vector store
            embedding_model = OpenAIEmbeddings(
                api_key=st.session_state['openai_api_key'], 
                model="text-embedding-3-small"
            )
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)

            # Setup retriever
            self.retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 2})

            # Setup prompt
            self.prompt = ChatPromptTemplate.from_template("""
Siz Najot Ta'lim o‘quv markazi uchun maxsus ishlab chiqilgan yordamchi chatbot sifatida quyidagi vazifalarni bajarishingiz kerak:

Asosiy vazifalar:
Markaz xizmatlari bo‘yicha ma’lumot berish: Kurslar, ularning davomiyligi, narxlari, jadvali, va o‘qituvchilar haqida aniq va batafsil ma’lumot taqdim etish.
O‘quvchilar savollariga javob berish: Foydalanuvchilarning tez-tez beriladigan savollariga, masalan, to‘lov usullari, chegirmalar, dars o‘tish formati (online/offline), hamda sertifikat olish tartibi haqida javob berish.
Hujjatga asoslangan javoblar: Faqat taqdim etilgan hujjatdagi ma’lumotlar asosida aniq va faktlarga mos javob berish. Agar kerakli javob hujjatda bo‘lmasa, foydalanuvchiga bu haqida xabar bering.
Foydalanuvchini yo‘naltirish: Kerak bo‘lganda foydalanuvchini muhim kontaktlar, manzil yoki veb-saytga yo‘naltirish.
Muhim qoidalar:
Faqat hujjatga asoslaning: Hujjatda keltirilmagan ma’lumotlarni o‘ylab topmang.
Do‘stona va professional ohangda bo‘ling: Har doim muloyim va yordamga tayyor bo‘ling.
Ma’lumotni aniq yetkazish: Javoblaringiz tushunarli, qisqa va mazmunli bo‘lishi kerak.
Keraksiz ma’lumot bermang: Faqat foydalanuvchi so‘ragan yoki hujjatda mavjud bo‘lgan ma’lumotni taqdim eting.
Tez-tez beriladigan savollarga misollar:
Kurslar va yo‘nalishlar
Qaysi kurslar mavjud?
Kurs narxlari qancha?
Darslar haftada necha kun o‘tiladi?, O‘qituvchilar haqida ma’lumot bera olasizmi?, To‘lov va chegirmalar, To‘lovni qanday amalga oshirish mumkin?, Chegirmalar bormi?, To‘lovni qismlarga bo‘lib to‘lash imkoniyati mavjudmi?, O‘qish jarayoni, Kurslar offline yoki online tarzda o‘tiladimi?, Kursni tugatgandan so‘ng sertifikat beriladimi?, Kurs davomiyligi qancha?, Qo‘shimcha ma’lumotlar , Markazning manzili qayerda?,Ish vaqtlari qanday?, Bog‘lanish uchun telefon raqamlari?
Javob namunasi:
Agar foydalanuvchi savol bersa, javobingiz quyidagicha bo‘lishi kerak:
Savol: "Kursi narxi qancha?"
Javob: "Hujjatga ko‘ra, Kursining narxi oyiga 2.400.000 so‘m. Bu kurs davomiyligi 11 oy bo‘lib, darslar haftada 5 marta 4.5 soatdan o‘tiladi. Qo‘shimcha ma’lumot uchun biz bilan bog‘lanishingiz mumkin. Bog'lanish +998-78-888-98-88"
            Context: {context}
            Question: {question}
            """)

            # Setup language model
            self.llm = ChatOpenAI(
                model="gpt-4",
                api_key=st.session_state['openai_api_key'],
                temperature=0.2
            )
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")




    def get_response(self, user_message: str) -> str:
        """Get response from the chatbot"""
        try:
            # Retrieve relevant documents
            retriever_output = self.retriever.invoke(user_message)
            context = " ".join([doc.page_content for doc in retriever_output])
        
            
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            return chain.invoke(user_message)
        except Exception as e:
            return f"Error generating response: {str(e)}"



    def display_messages(self):
        """Display chat messages"""
        for msg in st.session_state['messages']:
            if msg['role'] == 'user':
                with st.chat_message("user", avatar="👨‍💻"):
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg['content'])

    def setup_sidebar(self):
        """Configure sidebar with API key and chat management"""
        st.sidebar.title("Menu:")
        
        # API Key Input
        st.session_state['openai_api_key'] = st.sidebar.text_input(
            "OpenAI API Kalitni kiriting",
            type="password"
        )

        # New Chat Button
        if st.sidebar.button("Yangi chat"):
            self.start_new_chat()

        # # Chat History Section
        # st.sidebar.header("Chat Tarixi")
        # if st.session_state.get('chat_history', []):
        #     for idx, chat in enumerate(st.session_state['chat_history'], 1):
        #         st.sidebar.markdown(f"**Chat {idx}**")

    def start_new_chat(self):
        """Start a new chat and save previous chat"""
        if st.session_state['messages']:
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            st.session_state['chat_history'].append(st.session_state['messages'])
        st.session_state['messages'] = []
        st.session_state['thread_id'] = str(uuid.uuid4())

    def run(self):
        """Main application runner"""
        st.title("Najot Ta'lim ChatBot")

        # Sidebar setup
        self.setup_sidebar()

        # Check if API key is provided
        if not st.session_state['openai_api_key']:
            st.warning("Iltimos, OpenAI API kalitni kiriting.")
            return

        # Load PDF and vector store
        self.load_pdf_and_initialize_vectorstore()

        # Display existing messages
        self.display_messages()

        # Chat input
        if prompt := st.chat_input("Savolni kiriting"):
            # Add user message to session
            st.session_state['messages'].append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user", avatar="👨‍💻"):
                st.markdown(prompt)

            # Get and display AI response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("O'ylayapman..."):
                    response = self.get_response(prompt)
                    st.markdown(response)

            # Add AI response to session
            st.session_state['messages'].append({"role": "assistant", "content": response})

def main():
    chatbot = NajotTalimBot()
    chatbot.run()

if __name__ == "__main__":
    main()
