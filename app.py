import os
import markdown
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

# For structured output combined with typing 
from pydantic import BaseModel, Field 

# Flask : the application
# render_template : renders HTML docs
# redirect : reroutes to another route
# url_for : gets url for other route
# flash : flashes messages when something suceeds or fails
# request: handles details of the request
from flask import Flask, render_template, redirect, url_for, flash, request

from guardrails.sanitization import sanitize_input
from agent.orchestrator import run_agent
from observability.traces import instrumentor, tracer_provider
# Patches the OpenAI client so every chat.completions.create call is traced
# automatically – no manual span management needed in run_agent().
instrumentor.instrument(tracer_provider=tracer_provider)

app = Flask(__name__)
app.secret_key = 'secretkey-not-for-prod'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# GPT-4o returns markdown (**bold**, 1. item) but Jinja renders it as plain text. 
# You need to convert markdown to HTML before displaying it.
app.jinja_env.filters['markdown'] = lambda text: markdown.markdown(text)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Routes 

@app.route("/", methods=['GET', 'POST'])
def index():
    filename = None
    query = ""
    response = None

    if request.method == 'POST':
        raw_query = request.form.get("query", "")
        file = request.files.get('file')

        # Sanitize before anything else touches the query 
        query, error = sanitize_input(raw_query)
        if error:
            flash(error, "danger")
        
        if not query:
            flash("Please enter a search query", "danger")
            return redirect(url_for("index"))
        
        filepath = None
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            filename = file.filename

        try:
            # Prompt the model with tools defined
            # client.chat.completions.create = Chat Completions API
            # client.responses.create =  Responses API
            result = run_agent(query, filepath if file else None)
            if isinstance(result, dict):
                response = None
                products = result["products"]
            else:
                response = result  # general chat, no products
                products = []
            return render_template('index.html', 
                filename=filename, response=response, 
                products=products, query=query)
        except Exception as exc:
            flash(f"Agent error: {exc}", "danger")
            return redirect(url_for("index"))
        
    return render_template('index.html', filename=filename, response=response, query=query, products=[])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

# base.html have a base layout with the navigation and the header and everything.
# index.html extends base.html 