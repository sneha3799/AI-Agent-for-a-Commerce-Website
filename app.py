import os
import json
import asyncio
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
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
# from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

load_dotenv()
# llm = ChatOpenAI(model='gpt-4o')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

# server_params = StdioServerParameters(
#     command='npx',
#     args=['@brightdata/mcp'],
#     env={
#         'API_TOKEN': os.getenv('API_TOKEN'),
#         'WEB_UNLOCKER_ZONE': os.getenv('WEB_UNLOCKER_ZONE'),
#         'BROWSER_AUTH': os.getenv('BROWSER_AUTH')
#     }
# )

SYSTEM_PROMPT = (
    "To find products, first use the search_engine tool. When finding products, use the web_data tool for the platform. If none exists, scrape as markdown."
    "Example: Don't use web_data_bestbuy_products for search. Use it only for getting data on specific products you already found in search."
)

# PLATFORMS = ['Amazon', 'Best Buy', 'Ebay', 'Walmart', 'Target', 'Costco', 'Newegg']

# Defining data model using classes as we expect LLM output to have a certain structure

# For product hit
# class Hit(BaseModel):
#     url: str = Field(..., 'The URL of the product that was found')
#     title: str = Field(..., 'The title of the product that was found')
#     rating: str = Field(..., 'The rating of the product (stars, number of ratings given etc.)')

# # For each platform
# class PlatformBlock(BaseModel):
#     platform: str = Field(..., description='Name of the platform')
#     results: list[Hit] = Field(..., description='List of results for this platform')

# # Response object
# class ProductSearchResponse(BaseModel):
#     platforms: list[PlatformBlock] = Field(..., description='Aggregated list of all results grouped by platform')

app = Flask(__name__)
app.secret_key = 'secretkey-not-for-prod'

# LangGraph ReAct agent
# asynchronous function for running the agent
# async def run_agent(query, platforms):
#     # connect to MCP server
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as sess:
#             await sess.initialize()

#             tools = await load_mcp_tools(sess)
#             agent = create_react_agent(model, tools, response_format=ProductSearchResponse)

#             prompt = f'{query}\n\nPlatforms: {",".join(platforms)}'
#             result = await agent.invoke(
#                 {
#                     'messages': [
#                         {'role': 'system', 'content': SYSTEM_PROMPT},
#                         {'role': 'user', 'content': prompt}
#                     ]
#                 }
#             )
#             structured = result['structured_response']
#             return structured.model_dump()
        
# @app.route("/chat", methods=['GET', 'POST'])
# def chat():
#     if request.method=="POST":
#         query = request.form.get("query", "").strip()
#         image = request.form.getlist("image")

#         if not query:
#             flash("Please enter a search query", "danger")
#             return redirect(url_for("index"))

#         # if not platforms:
#         #     flash("Select atleast one platform", "danger")
#         #     return redirect(url_for("index"))
        
#         try:
#             response_json = asyncio.run(run_agent(query, image))
#         except Exception as exc:
#             flash(f"Agent error: {exc}", "danger")
#             return redirect(url_for("index"))
        
#         return render_template(
#             "index.html",
#             query= query, 
#             image = image,
#             response = response_json
#         )
    
#     return render_template(
#         "index.html",
#         query= "", 
#         image = None,
#         response = None
#     )

def generate_embeddings(input, is_image=False):
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
 
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method=="POST":
        query = request.form.get("query", "").strip()
        platforms = request.form.getlist("platforms")

        if not query:
            flash("Please enter a search query", "danger")
            return redirect(url_for("index"))

        if not platforms:
            flash("Select atleast one platform", "danger")
            return redirect(url_for("index"))
        
        try:
            response_json = asyncio.run(run_agent(query, platforms))
        except Exception as exc:
            flash(f"Agent error: {exc}", "danger")
            return redirect(url_for("index"))
        
        return render_template(
            "index.html",
            query= query, 
            platforms= PLATFORMS,
            selected= platforms,
            response = response_json
        )
    
    return render_template(
        "index.html",
        query= "", 
        platforms= PLATFORMS,
        selected= [],
        response = None
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

# base.html have a base layout with the navigation and the header and everything.
# index.html extends base.html 