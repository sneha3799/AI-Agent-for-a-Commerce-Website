import psycopg2
import os
from strands import tool

from dotenv import load_dotenv
load_dotenv()

from retrieval.embedder import generate_embeddings

def get_connection():
    """Fresh connection per call — avoids closed cursor issues"""
    return psycopg2.connect(os.getenv('URL'))

QUERY = """
    SELECT id, product_display_name, image_name, master_category, base_colour
    FROM products
    ORDER BY embedding <=> %s::vector
    LIMIT 5
"""

def rows_to_dicts(rows: list) -> list:
    return [
        {
            "id": row[0],
            "product_display_name": row[1],
            "image_name": row[2],
            "master_category": row[3],
            "base_colour": row[4]
        }
        for row in rows
    ]

# Define a custom tool as a python function using the tool decorator
# The docstring matters — Strands uses it as the tool description for the LLM, 
# just like JSON tool definitions did with OpenAI.
# Tool functions must return strings, not lists

@tool
def product_recommendation(query: str) -> list:
    """Search products by text description. Use when the user asks for product recommendations."""
    print(f"🔧 TOOL CALLED with: {query}")
    embedding = generate_embeddings(query, is_image=False)
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(QUERY, (embedding, ))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        results = rows_to_dicts(rows)
        print('--', results)
        return results
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        return []

@tool
def image_product_search(image):
    """Search for similar products using an image. Use when the user uploads an image."""
    embedding = generate_embeddings(image)
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(QUERY, (embedding,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        results = rows_to_dicts(rows)
        print(f"🔧 RETURNED {len(results)} results: {results[:1]}")
        return results
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        return []