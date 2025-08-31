import pyttsx3
import pygame
from gtts import gTTS
import os
import tempfile
import threading
import time

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Language configuration
LANGUAGES = {
    'English': {
        'code': 'en',
        'gtts_lang': 'en',
        'pyttsx3_voice': 'english'
    },
    'Hindi': {
        'code': 'hi', 
        'gtts_lang': 'hi',
        'pyttsx3_voice': 'hindi'
    },
    'Tamil': {
        'code': 'ta',
        'gtts_lang': 'ta', 
        'pyttsx3_voice': 'tamil'
    }
}

# Global language setting
current_language = 'English'

def set_language(language):
    """Set the current language for voice alerts"""
    global current_language
    if language in LANGUAGES:
        current_language = language
        print(f"Language set to: {language}")
    else:
        print(f"Language {language} not supported. Using English.")
        current_language = 'English'

def speak_alert_pyttsx3(text):
    """Use pyttsx3 for voice synthesis (works offline)"""
    try:
        engine = pyttsx3.init()
        
        # Set voice properties based on language
        voices = engine.getProperty('voices')
        lang_config = LANGUAGES[current_language]
        
        # Try to find a voice for the selected language
        selected_voice = None
        for voice in voices:
            if lang_config['code'] in voice.id.lower() or lang_config['pyttsx3_voice'] in voice.name.lower():
                selected_voice = voice.id
                break
        
        if selected_voice:
            engine.setProperty('voice', selected_voice)
        
        # Set speech rate and volume
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
    except Exception as e:
        print(f"Error with pyttsx3: {e}")
        # Fallback to default voice
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()

def speak_alert_gtts(text):
    """Use gTTS for voice synthesis (requires internet)"""
    try:
        lang_config = LANGUAGES[current_language]
        
        # Create gTTS object
        tts = gTTS(text=text, lang=lang_config['gtts_lang'], slow=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)
        
        # Play the audio file using pygame
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Clean up temporary file
        try:
            os.unlink(temp_filename)
        except:
            pass
            
    except Exception as e:
        print(f"Error with gTTS: {e}")
        # Fallback to pyttsx3
        speak_alert_pyttsx3(text)

def speak_alert(text, use_gtts=True):
    """
    Main function to speak alerts in the selected language
    Args:
        text: Text to speak
        use_gtts: If True, try gTTS first, then fallback to pyttsx3
    """
    try:
        # Translate text based on language if needed
        translated_text = translate_text(text)
        
        if use_gtts:
            speak_alert_gtts(translated_text)
        else:
            speak_alert_pyttsx3(translated_text)
            
    except Exception as e:
        print(f"Error in speak_alert: {e}")
        # Ultimate fallback - English with pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except:
            print("All speech synthesis methods failed")

def translate_text(text):
    """Translate common phrases to selected language"""
    translations = {
        'English': {
            'detected. Please be careful.': 'detected. Please be careful.',
            'Road sign detected. Please be careful.': 'Road sign detected. Please be careful.'
        },
        'Hindi': {
            'detected. Please be careful.': 'मिला। कृपया सावधान रहें।',
            'Road sign detected. Please be careful.': 'सड़क का संकेत मिला। कृपया सावधान रहें।',
            'Speed limit': 'गति सीमा',
            'Stop': 'रुको',
            'No entry': 'प्रवेश नहीं',
            'Turn left': 'बाएं मुड़ें',
            'Turn right': 'दाएं मुड़ें',
            'No parking': 'पार्किंग नहीं'
        },
        'Tamil': {
            'detected. Please be careful.': 'கண்டறியப்பட்டது। தயவுசெய்து கவனமாக இருங்கள்.',
            'Road sign detected. Please be careful.': 'சாலை அடையாளம் கண்டறியப்பட்டது। தயவுசெய்து கவனமாக இருங்கள்.',
            'Speed limit': 'வேக வரம்பு',
            'Stop': 'நிறுத்து',
            'No entry': 'நுழைவு இல்லை',
            'Turn left': 'இடது திருப்பு',
            'Turn right': 'வலது திருப்பு',
            'No parking': 'பார்க்கிங் இல்லை'
        }
    }
    
    # Get translations for current language
    lang_translations = translations.get(current_language, translations['English'])
    
    # Translate the text
    for english_phrase, translated_phrase in lang_translations.items():
        if english_phrase in text:
            text = text.replace(english_phrase, translated_phrase)
    
    return text

def get_available_languages():
    """Return list of available languages"""
    return list(LANGUAGES.keys())

def get_current_language():
    """Return current selected language"""
    return current_language