# Phoenix is an application that can receive the traces that you're going to send from your agent here and then can visualize those in a UI

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from openinference.instrumentation import TracerProvider

from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry import trace
import os
from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ before any key checks

import nest_asyncio # for running multiple calls asynchronously and simultaneously to speed up some of your evaluation
nest_asyncio.apply()

# Phoenix is an application that can receive the traces that you're going to send from your agent here and then can visualize those in a UI
import phoenix as px
px_client = px.Client()

# Add Phoenix API Key for tracing — load from environment, never hardcode secrets
os.environ.setdefault("PHOENIX_API_KEY", os.getenv("PHOENIX_API_KEY", ""))
os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/s/CaVe-VLM-CoT"))
_api_key = os.environ["PHOENIX_API_KEY"]
if not _api_key:
    raise EnvironmentError(
        "PHOENIX_API_KEY is not set. Export it before running: "
        "export PHOENIX_API_KEY=<your-key>"
    )
# If you created your Phoenix Cloud instance before June 24th, 2025,
# you also need to set the API key as a header
# os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"

# Start Phoenix server
session = px.launch_app()

PROJECT_NAME = "cite-and-verify-vlm-cot-agent"
# Register base provider (phoenix sets a SimpleSpanProcessor by default;
# we immediately replace it with a BatchSpanProcessor so spans are
# buffered and exported in bulk, avoiding one blocking HTTP call per span).
tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces",
    headers={
        "Authorization": f"Bearer {os.environ['PHOENIX_API_KEY']}"
    }
)
# Swap in a BatchSpanProcessor.
# register() warns that add_span_processor overwrites its default —
# that's exactly what we want here.
_batch_exporter = OTLPSpanExporter(
    endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces",
    headers={"authorization": f"Bearer {os.environ['PHOENIX_API_KEY']}"},
)
tracer_provider.add_span_processor(
    BatchSpanProcessor(
        _batch_exporter,
        max_queue_size=2048,       # buffer up to 2048 spans before dropping
        schedule_delay_millis=5000, # flush every 5 s (not every span)
        max_export_batch_size=512,  # send up to 512 spans per HTTP call
    )
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)