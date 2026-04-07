# Input sanitization 
import re
import logging

logger = logging.getLogger(__name__)

# Phrases that are strong indicators of prompt-injection attempts.
# Keeping the list explicit (rather than purely regex) makes it easy to audit
# and extend without accidentally over-blocking legitimate fashion queries.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(?:a|an|the)\s+\w+",   # "you are now a pirate / DAN / …"
    r"act\s+as\s+(?:a|an|the)\s+\w+",
    r"do\s+not\s+follow\s+your\s+(system\s+)?prompt",
    r"reveal\s+your\s+(system\s+)?prompt",
    r"print\s+your\s+(system\s+)?prompt",
    r"override\s+(your\s+)?(instructions?|rules?|guidelines?)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"<\s*script[^>]*>",                        # HTML/JS injection
    r"\{\{.*?\}\}",                             # Template injection  {{ ... }}
    r"\$\{.*?\}",                               # Template injection  ${ ... }
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
MAX_QUERY_LENGTH = 500  # characters; generous for a product search query

def sanitize_input(text: str) -> tuple[str, str | None]:
    """
    Clean and validate user-supplied query text before it reaches the LLM.

    Returns:
        (cleaned_text, error_message)
        If error_message is not None the input should be rejected and the
        message shown to the user.

    Sanitization steps
    1. Strip leading/trailing whitespace.
    2. Remove ASCII control characters (except newline/tab) – these can
       confuse tokenisers and are never needed in a product query.
    3. Enforce a maximum length to bound token cost and surface suspiciously
       long payloads.
    4. Scan for injection trigger phrases and bail out if any match.
    """
    if not text:
        return "", "Query cannot be empty."

    # Step 1 – strip whitespace
    text = text.strip()

    # Step 2 – remove ASCII control characters (0x00–0x1F) except \t and \n
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Step 3 – length cap
    if len(text) > MAX_QUERY_LENGTH:
        return "", (
            f"Query is too long ({len(text)} chars). "
            f"Please keep it under {MAX_QUERY_LENGTH} characters."
        )

    # Step 4 – injection pattern check
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            # Log the raw attempt for your own monitoring; do NOT echo it back
            # to the user (that can itself leak information about the filter).
            logger.warning("Prompt injection attempt blocked: %r", text[:120])
            return "", (
                "Your query contains content that cannot be processed. "
                "Please describe the product you are looking for."
            )

    return text, None