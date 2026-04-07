import psycopg2
import os

from dotenv import load_dotenv
load_dotenv()

from retrieval.embedder import generate_embeddings

url = os.getenv('URL')
def product_recommendation(query):
    embedding = generate_embeddings(query, is_image=False)
    
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    results = []
    try:
        cur.execute("""
            SELECT id, product_display_name, image_name, master_category, base_colour
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
    embedding = generate_embeddings(image)
    
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    results = []
    try:
        cur.execute("""
            SELECT id, product_display_name, image_name, master_category, base_colour
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