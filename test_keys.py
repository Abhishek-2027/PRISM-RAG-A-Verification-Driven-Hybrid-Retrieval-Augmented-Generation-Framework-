import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

load_dotenv()

def test_groq():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("[FAIL] GROQ_API_KEY not found in .env")
        return False
    try:
        # Using a currently active model
        llm = ChatGroq(groq_api_key=key, model_name="llama-3.3-70b-versatile")
        res = llm.invoke("Hi")
        print("[PASS] Groq Key is working (with llama-3.3-70b-versatile)!")
        return True
    except Exception as e:
        print(f"[FAIL] Groq Key failed: {str(e)}")
        return False

def test_gemini():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("[FAIL] GOOGLE_API_KEY not found in .env")
        return False
    try:
        llm = ChatGoogleGenerativeAI(google_api_key=key, model="gemini-1.5-flash")
        res = llm.invoke("Hi")
        print("[PASS] Gemini Key is working (with gemini-1.5-flash)!")
        return True
    except Exception as e:
        print(f"[FAIL] Gemini Key failed: {str(e)}")
        return False

def test_langchain():
    key = os.getenv("LANGCHAIN_API_KEY")
    if not key:
        print("[FAIL] LANGCHAIN_API_KEY not found in .env")
        return False
    if key.startswith("lsv2_pt_"):
        print("[PASS] LangChain Key format looks valid!")
        return True
    else:
        print("[FAIL] LangChain Key format looks invalid.")
        return False

if __name__ == "__main__":
    print("Testing API Keys...")
    groq_ok = test_groq()
    gemini_ok = test_gemini()
    langchain_ok = test_langchain()
    
    if groq_ok and gemini_ok and langchain_ok:
        print("\nAll keys verified successfully!")
    else:
        print("\nSome keys failed. Please check your .env file.")
