# Do not modify !
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
hf_model="mistralai/Mixtral-8x7B-Instruct-v0.1"