# Documentation - https://strandsagents.com/docs/user-guide/concepts/agents/structured-output/

import os
import base64
import imghdr

from strands import Agent
from strands.models import BedrockModel
from agent.tools import product_recommendation, image_product_search
from pydantic import BaseModel, Field
from typing import List

# Define the model
model_id = BedrockModel(
    # model_id="us.anthropic.claude-sonnet-4-6",
    # model_id="us.amazon.nova-lite-v1:0",  # lighter model, higher limits
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", 
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Define the system prompt
system_prompt = """
You are an ecommerce assistant for a fashion store.

STRICT RULES:
- For ANY product, clothing, or shopping query: MUST call product_recommendation tool.
- For image uploads: MUST call image_product_search tool.
- NEVER invent product names, IDs, or image filenames.
- Only return products that come directly from tool results.
- For greetings or general questions: respond in text field, leave products as [].
"""

# Define the agent
agent = Agent(
    model=model_id,
    system_prompt=system_prompt,
    tools=[
        product_recommendation,
        image_product_search,
    ],
)

# Define a custom output structure using Pydantic models
class ProductDetails(BaseModel):
    """Product Details"""
    text: str = Field(description="Model response text")
    products: List[dict] = Field(
        default=[],
        description="Product details such as id, product_display_name, image_name, master_category, base_colour"
    )

# Run the agent
def run_agent(query, image_path=None):
    if image_path:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            # base64-encoded image data
            # image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect actual format instead of hardcoding "png"
        fmt = imghdr.what(None, h=image_bytes) or "jpeg"
        # imghdr returns "jpeg" but Bedrock expects "jpeg" — normalize just in case
        fmt = fmt.replace("jpg", "jpeg")
    
        # https://levelup.gitconnected.com/strands-agent-key-features-i-use-like-the-most-20b280d265cd
        response = agent(
            [
                {"text": query}, 
                {"image": {"format": fmt, "source": {"bytes": image_bytes}}}
            ],
            structured_output_model=ProductDetails
        )
    else:
        response = agent(
            query,
            structured_output_model=ProductDetails
        )

    # response type <class 'strands.agent.agent_result.AgentResult'>
    # structured output type <class 'agent.strands_agent.ProductDetails'>
    return {
        "text": response.structured_output.text,
        "products": 
        [
            {
                "id": p.get("id"),
                "name": p.get("product_display_name"),
                "image": os.path.basename(p.get("image_name", "")),
                "category": p.get("master_category"),
                "colour": p.get("base_colour")
            }
            for p in (response.structured_output.products or [])
        ]
    }