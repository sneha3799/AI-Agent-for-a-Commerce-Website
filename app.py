import os
import json
import asyncio
import psycopg2
import base64
from typing import List, Optional

from dotenv import load_dotenv

# For structured output combined with typing 
from pydantic import BaseModel, Field 

import open_clip
import torch
from PIL import Image

# Flask : the application
# render_template : renders HTML docs
# redirect : reroutes to another route
# url_for : gets url for other route
# flash : flashes messages when something suceeds or fails
# request: handles details of the request
from flask import Flask, render_template, redirect, url_for, flash, request

load_dotenv()

# from langchain_openai import ChatOpenAI, custom_tool
# from langchain.agents import create_agent
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

url = os.getenv('URL')

app = Flask(__name__)
app.secret_key = 'secretkey-not-for-prod'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# LangGraph ReAct agent
# asynchronous function for running the agent

def generate_embeddings(input, is_image=True):
    if is_image:
        image = Image.open(input).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
    else:
        text_input = tokenizer([input]).to(device)
        with torch.no_grad():
            embedding = model.encode_text(text_input)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().tolist()

# Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "product_recommendation",
            "description": "Text-Based Product Recommendation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question around product recommendations",
                    },
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_product_search",
            "description": "Image-Based Product Search",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "path to the image file",
                    },
                },
                "required": ["file"],
            },
        },
    }
]

def product_recommendation(query):
    embedding = generate_embeddings(query, is_image=False)
    
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    results = []
    try:
        cur.execute("""
            SELECT id, product_display_name
            FROM products
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (embedding, ))
        results = cur.fetchall()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

    cur.close()
    conn.close()
    return results

def image_product_search(image):
    # if query is None:
    embedding = generate_embeddings(image)
    # elif image is None:
    #     embedding = generate_embeddings(query, is_image=False)
    # else:
    #     image_vector = generate_embeddings(image)
    #     text_vector = generate_embeddings(query, is_image=False)
    #     embedding = [(a + b) / 2 for a, b in zip(text_vector, image_vector)]
    
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    results = []
    try:
        cur.execute("""
            SELECT id, product_display_name
            FROM products
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        """, (embedding, ))
        results = cur.fetchall()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()

    cur.close()
    conn.close()
    return results

# Create a running input list we will add to over time
input_list = [
    {"role": "system", "content": "You are a ecommerce search assistant to help customers get the right product based on their needs."}
]

def run_agent(query, image_path=None):
    messages = [
        {"role": "system", "content": "You are an ecommerce assistant. Use the tools to search products. When the user provides an image, use image_product_search to find similar products."},
    ]

    # Build user message based on whether image is provided
    if image_path:
        with open(image_path, "rb") as f:
            # sends the image inline so GPT-4o can see it, and it will then call image_product_search. Your CLIP function separately embeds the image from image_path for the pgvector search.
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": query}
        ]
    else:
        user_content = query
    
    messages.append({"role": "user", "content": user_content})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    print(f'Response: {response}')
    message = response.choices[0].message
    
    # If no tool call → general conversation, return directly
    if not message.tool_calls:
        return message.content
    
    # Tool was called → execute it
    messages.append(message)  # add assistant's tool call to history
    
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == "product_recommendation":
            result = product_recommendation(args["query"])
        elif tool_call.function.name == "image_product_search":
            result = image_product_search(image_path)
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })
    
    # Send results back to GPT-4o for a natural language response
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    return final.choices[0].message.content

@app.route("/", methods=['GET', 'POST'])
def index():
    filename = None
    query = ""
    response = None

    if request.method == 'POST':
        query = request.form.get("query", "").strip()
        file = request.files.get('file')
        
        if not query:
            flash("Please enter a search query", "danger")
            return redirect(url_for("index"))
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],  
                                    file.filename)
            file.save(filepath)
            filename = file.filename

        try:
            # Prompt the model with tools defined
            # client.chat.completions.create = Chat Completions API
            # client.responses.create =  Responses API
            response = run_agent(query, filepath if file else None)
            print(f'Results: {response}')
        except Exception as exc:
            flash(f"Agent error: {exc}", "danger")
            return redirect(url_for("index"))
        
    return render_template('index.html', filename=filename, response=response, query=query)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

# base.html have a base layout with the navigation and the header and everything.
# index.html extends base.html 