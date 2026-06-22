# Capture traces and spans in Arize Phoenix
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import os

# register() sets up an OTLP exporter that points at your local (or hosted)
# Phoenix collector.  Set PHOENIX_COLLECTOR_ENDPOINT in .env if your Phoenix
# server is not on the default http://localhost:6006.
tracer_provider = register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "ecommerce-agent"),
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
    headers={"api_key": os.getenv("PHOENIX_API_KEY", "")},
)

instrumentor = OpenAIInstrumentor()