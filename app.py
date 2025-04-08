import streamlit as st
from openai import OpenAI
from PIL import Image
import io
import os
import time
from dotenv import load_dotenv
from openai import OpenAIError, AuthenticationError, RateLimitError, APIError

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Chatbot & Image Generator",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history and API key
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY")
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# Initialize OpenAI client
client = None
if st.session_state.api_key:
    client = OpenAI(api_key=st.session_state.api_key)

# Sidebar for API key input
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.api_key or "")
    if api_key and api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        client = OpenAI(api_key=api_key)
        st.success("API key updated successfully!")

# Check if API key is set
if not st.session_state.api_key:
    st.error("Please enter your OpenAI API key in the sidebar to use the application.")
    st.stop()

def make_request_with_retry(func, *args, max_retries=3, **kwargs):
    """Helper function to make API requests with retry logic"""
    for attempt in range(max_retries):
        try:
            # Add delay between requests to avoid rate limits
            current_time = time.time()
            time_since_last_request = current_time - st.session_state.last_request_time
            if time_since_last_request < 1:  # Wait at least 1 second between requests
                time.sleep(1 - time_since_last_request)
            
            result = func(*args, **kwargs)
            st.session_state.last_request_time = time.time()
            return result
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = (attempt + 1) * 5  # Exponential backoff
            st.warning(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            raise e

# Main content
st.title("🤖 AI Chatbot & Image Generator")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Chat", "Image Generation"])

# Chat tab
with tab1:
    st.header("Chat with AI")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                def get_chat_response():
                    return client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )
                
                response = make_request_with_retry(get_chat_response)
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except AuthenticationError:
                st.error("Invalid API key. Please check your API key in the sidebar.")
            except RateLimitError:
                st.error("Rate limit exceeded. Please wait a few minutes and try again.")
            except APIError as e:
                st.error(f"OpenAI API error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Image Generation tab
with tab2:
    st.header("Generate Images")
    
    # Image generation input
    image_prompt = st.text_input("Describe the image you want to generate")
    
    if st.button("Generate Image"):
        if not image_prompt:
            st.warning("Please enter a description for the image")
        else:
            try:
                with st.spinner("Generating image..."):
                    def get_image_response():
                        return client.images.generate(
                            prompt=image_prompt,
                            n=1,
                            size="1024x1024"
                        )
                    
                    response = make_request_with_retry(get_image_response)
                    image_url = response.data[0].url
                    st.image(image_url, caption=image_prompt)
                    
            except AuthenticationError:
                st.error("Invalid API key. Please check your API key in the sidebar.")
            except RateLimitError:
                st.error("Rate limit exceeded. Please wait a few minutes and try again.")
            except APIError as e:
                st.error(f"OpenAI API error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using OpenAI's GPT-3.5 and DALL-E") 