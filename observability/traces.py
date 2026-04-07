# Capture traces and spans in Arize Phoenix
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# register() sets up an OTLP exporter that points at your local (or hosted)
# Phoenix collector.  Set PHOENIX_COLLECTOR_ENDPOINT in .env if your Phoenix
# server is not on the default http://localhost:6006.
tracer_provider = register(
  project_name="ecommerce-agent",
  auto_instrument=True
)

instrumentor = OpenAIInstrumentor()