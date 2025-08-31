# Voice alert package initialization
from .tts_engine import speak_alert, set_language, get_available_languages, get_current_language

__all__ = ['speak_alert', 'set_language', 'get_available_languages', 'get_current_language']