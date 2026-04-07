# Agent
import base64
import os
import json 

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

from agent.tools import product_recommendation, image_product_search

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

def run_agent(query, image_path=None):
    """
    Two-turn ReAct loop:
      Turn 1 → GPT-4o decides which tool to call (or replies directly).
      Turn 2 → GPT-4o sees the tool result and produces the final answer.

    Both turns are automatically captured as OpenTelemetry spans by the
    OpenAIInstrumentor registered at startup, so they appear in Phoenix
    under the project name set in PHOENIX_PROJECT_NAME.
    """

    # Create a running input list we will add to over time
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
    
    # Turn 1 – tool dispatch
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
    
    found_products = []
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        
        if tool_call.function.name == "product_recommendation":
            result = product_recommendation(args["query"])
            found_products = result
        elif tool_call.function.name == "image_product_search":
            result = image_product_search(image_path)
            found_products = result
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })
    
    # Turn 2 – natural language answer
    # Send results back to GPT-4o for a natural language response
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    return {
        "text": final.choices[0].message.content,
        "products": [
            {
                "id": p[0],
                "name": p[1],
                "image": os.path.basename(p[2]),  # turns "/Users/.../static/images/12345.jpg" into "12345.jpg"
                "category": p[3],
                "colour": p[4]
            }
            for p in found_products
        ]
    }