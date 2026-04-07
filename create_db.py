import psycopg2 # the most popular PostgreSQL database adapter for Python
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

from tqdm import tqdm
tqdm.pandas()

from dotenv import load_dotenv
load_dotenv()
db_url = os.getenv("DATABASE_URL")
url = os.getenv('URL')

from retrieval.embedder import generate_embeddings

# Use os.path.expanduser to handle '~'
base_dir = Path(__file__).parent  # assumes script is in palona_ai_agent/
image_dir = base_dir / 'static' / 'images'
csv_path = base_dir / 'myntradataset' / 'styles.csv'

engine = create_engine(db_url)

# The connect() method from the psycopg2 module returns a Connection object, 
# when a connection object is created it essentially sets up a client session 
# with the database server. This session is persistent, meaning it remains open 
# until you explicitly close it. This persistence allows you to send multiple queries
#  and commands to the database without needing to establish a new connection each time. So, you have a continuous line of communication between your Python code and the database server, facilitated by this persistent client session.
conn = psycopg2.connect(url)

# Cursor object acts as a pointer or handler that allows you to execute SQL commands
#  within your established database connection. It essentially bridges the gap
#  between your Python script and the database server, enabling you to perform
#  database operations.
cur = conn.cursor()

# Using the execute() function on the Cursor object with a stringified SQL query, 
# you can execute commands on the Postgres database.
# one = cur.fetchone()
# allresults = cur.fetchall()
cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector
""")

try:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS products(
            id integer PRIMARY KEY,
            gender text,
            master_category text,
            sub_category text,
            article_type text,
            base_colour text,
            season text,
            year double precision,
            usage text,
            product_display_name text,
            image_name text,
            embedding vector(512)
        )
    """)

    conn.commit()
    print("Table created successfully!")
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()

# Get and filter files
filenames = [f for f in os.listdir(image_dir)]

# Create DataFrame and save
df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
df['imageName'] = df['id'].apply(lambda x: str(image_dir / f"{x}.jpg"))

def safe_generate_embeddings(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        return generate_embeddings(image_path, is_image=True)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

df['embeddings'] = df['imageName'].progress_apply(safe_generate_embeddings)

original_count = len(df)
# Drop rows where embedding failed
df = df.dropna(subset=['embeddings'])
print(f"{len(df)} images processed successfully")
print(f"Kept {len(df)} / {original_count} products ({original_count - len(df)} missing images)")

df = df.rename(columns={
    'masterCategory': 'master_category',
    'subCategory': 'sub_category',
    'articleType': 'article_type',
    'baseColour': 'base_colour',
    'productDisplayName': 'product_display_name',
    'imageName': 'image_name',       
    'embeddings': 'embedding' 
})

keep_columns = ['id', 'gender', 'master_category', 'sub_category', 'article_type',
                'base_colour', 'season', 'year', 'usage', 'product_display_name',
                'image_name', 'embedding']
df = df[keep_columns]

# Perform the data loading operation
# With the to_sql() method, pandas handles the data transfer efficiently, leveraging PostgreSQL's capabilities in the background. 
# Then append data (preserves your schema)
df.to_sql('products', engine, if_exists='append', index=False)

# commit the changes to persist the data in the database and close the database connection when you're done.
conn.commit()
#Close the connection
conn.close()