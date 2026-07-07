"""Intent detection — first stage of the Chat Flow in the brief."""
import logging
from typing import Dict, List, Tuple

from app.llm_client import BaseLLMClient, LLMError
from app.prompts import ALFRED_PERSONA, INTENT_CLASSIFIER_PROMPT
from app.schemas import Intent

logger = logging.getLogger("alfred.intent")

_VALID_INTENTS = {i.value for i in Intent}


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(no prior messages)"
    return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history[-6:])


async def detect_intent(llm: BaseLLMClient, message: str, history: List[Dict[str, str]]) -> Tuple[Intent, float]:
    prompt = INTENT_CLASSIFIER_PROMPT.format(message=message, history=_format_history(history))
    try:
        data = await llm.complete_json(ALFRED_PERSONA, prompt)
        intent_str = str(data.get("intent", "general_chat"))
        confidence = float(data.get("confidence", 0.6))
    except (LLMError, ValueError, TypeError) as exc:
        logger.warning("Intent detection fell back to general_chat: %s", exc)
        return Intent.general_chat, 0.4

    if intent_str not in _VALID_INTENTS:
        logger.warning("Model returned unknown intent %r, defaulting to general_chat", intent_str)
        return Intent.general_chat, min(confidence, 0.5)

    return Intent(intent_str), max(0.0, min(confidence, 1.0))
