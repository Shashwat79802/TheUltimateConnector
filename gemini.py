import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
import os

dotenv.load_dotenv('.env')

# GEMINI_MODEL = ChatGoogleGenerativeAI(
#     api_key=os.getenv('GEMINI_API_KEY'),
#     google_api_key="gemini-1.5-flash",
#     verbose=True,
#     temperature=0.6
# )

GEMINI_MODEL = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash', verbose=True, temperature=0.9, google_api_key=os.getenv('GEMINI_API_KEY')
)

# response = GEMINI_MODEL.generate_content("Hii how are you")

# genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
#
# GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# response = GEMINI_MODEL.generate_content("Hii how are you")
#
# print(response.text)
