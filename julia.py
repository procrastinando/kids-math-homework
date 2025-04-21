import subprocess
import sys

required_packages = [
    "groq",
    "gradio",
    "gTTS",
    "requests",
    "regex",
    "jieba",
    "PyYAML"  # PyYAML provides the yaml module
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in required_packages:
    try:
        # Special handling for PyYAML which is imported as 'yaml'
        if package == "PyYAML":
            import yaml
        else:
            __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        install(package)

# Your script's imports
import gradio as gr
from gtts import gTTS
import requests
import tempfile
import os
import re
import difflib
import regex
import sys
import jieba
import yaml
import random
from fractions import Fraction
import datetime

# Mapping of user-friendly language names to gTTS and Groq Whisper API language codes
LANGUAGE_CODES = {
    'English': {'gtts': 'en', 'whisper': 'en'},
    'Spanish': {'gtts': 'es', 'whisper': 'es'},
    'Chinese': {'gtts': 'zh', 'whisper': 'zh'},
    'Russian': {'gtts': 'ru', 'whisper': 'ru'}
}

# Your Groq API Key (Replace this with your actual API key)
GROQ_API_KEY = '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'

def generate_tts(language, poem_text, line_range):
    """
    Generates TTS audio for the selected lines of the poem.

    Parameters:
    - language (str): Selected language.
    - poem_text (str): The full poem text.
    - line_range (tuple): Tuple containing start and end line numbers.

    Returns:
    - audio_file (temp file): Generated TTS audio file.
    - error (str): Error message if any.
    """
    try:
        # Split the poem into individual lines
        lines = poem_text.strip().split('\n')
        total_lines = len(lines)

        # Extract start and end lines from the range
        start, end = line_range

        # Validate the line range
        if total_lines < end:
            end = total_lines
        if start > end:
            start = end

        # Select the specified lines
        selected_lines = lines[start-1:end]
        tts_text = '\n'.join(selected_lines)

        # Initialize gTTS with the appropriate language code
        tts = gTTS(text=tts_text, lang=LANGUAGE_CODES[language]['gtts'])

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_file = fp.name
            tts.save(temp_file)

        return temp_file, None

    except Exception as e:
        return None, str(e)

def preprocess(text, language=''):
    """
    Remove special characters and punctuation, normalize whitespace,
    and convert text to lowercase. Supports multiple languages.
    """
    if language.lower() == 'chinese':
        # Perform word segmentation for Chinese text using jieba
        tokens = jieba.cut(text)
        text = ' '.join(tokens)

    # Normalize whitespace characters (space, tab, newline) to a single space
    text = regex.sub(r'\s+', ' ', text)

    # Remove punctuation and special characters
    # This regex retains Unicode letters and numbers
    text = regex.sub(r'[^\p{L}\p{N}\s]', '', text)

    # Convert to lowercase for case-insensitive comparison (irrelevant for Chinese)
    text = text.lower()

    return text.strip()

def transcribe_and_score(audio_path, language, poem_text):
    """
    Transcribes the user's audio using Groq Whisper API and computes a similarity score.
    
    Parameters:
    - audio_path (str): Path to the user's audio file.
    - language (str): Selected language.
    - poem_text (str): The full poem text.
    
    Returns:
    - transcription (str): Transcribed text from the audio.
    - score (str): Similarity score as a percentage.
    """
    try:
        if audio_path is None:
            return "No audio provided.", "0%"

        # Initialize the Groq client with your API key
        client = Groq(api_key=GROQ_API_KEY)

        # Specify the path to the audio file
        filename = audio_path  # Replace with your audio file path

        # Open the audio file
        with open(filename, "rb") as file:
            # Create a transcription of the audio file
            transcription_response = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Required audio file
                model="whisper-large-v3-turbo",  # Required model to use for transcription
                language=LANGUAGE_CODES[language]['whisper'],  # Optional
                response_format="json",  # Optional, default is json
                temperature=0.0  # Optional
            )
            
            # Extract the transcription text
            transcription = transcription_response.text.strip()

            if not transcription:
                return "Transcription failed.", "0%"

            # Preprocess the strings
            clean_str1 = preprocess(poem_text, language)
            clean_str2 = preprocess(transcription, language)
            # Create a SequenceMatcher object
            matcher = difflib.SequenceMatcher(None, clean_str1, clean_str2)
            # Get the similarity ratio and convert to percentage
            similarity = round((matcher.ratio() * 100)**2, 0)

            return transcription, f"{similarity}%"

    except Exception as e:
        return f"An error occurred: {e}", "0%"

### Mathematics functions

def generate_math_question(q_type):
    
    if q_type == 1:
        # Multiplication: Bags of apples
        X = random.randint(9, 99)
        Y = random.randint(9, 99)
        question = f"There are {X} bags and each bag contains {Y} apples. How many apples are there in total?"
        answer = str(X * Y)
        
    elif q_type == 2:
        # Division: Exact division problem
        B = random.randint(20, 99)
        quotient = random.randint(2, 9)
        A = B * quotient  # ensures no remainder
        question = f"Divide {A} by {B}. What is the quotient?"
        answer = str(quotient)
        
    elif q_type == 3:
        # Multi-digit addition
        A = random.randint(100, 999)
        B = random.randint(100, 999)
        question = f"Add {A} and {B}."
        answer = str(A + B)
        
    elif q_type == 4:
        # Multi-digit subtraction (ensure A > B)
        A = random.randint(100, 999)
        B = random.randint(10, A - 1)
        question = f"Subtract {B} from {A}."
        answer = str(A - B)
        
    elif q_type == 5:
        # Fraction: What is 1/N of X? (X is a multiple of N)
        N = random.choice([2, 3, 4, 5, 6, 8, 10])
        quotient = random.randint(20, 200)
        X = N * quotient
        question = f"What is 1/{N} of {X}?"
        answer = str(quotient)
        
    elif q_type == 6:
        # Word problem: Equal groups in a classroom
        groups = random.randint(2, 10)
        students_per_group = random.randint(5, 30)
        total_students = groups * students_per_group
        question = (f"A classroom has {total_students} students divided equally into {groups} groups. "
                    "How many students are in each group?")
        answer = str(students_per_group)
        
    elif q_type == 7:
        # Area of a rectangle
        length = random.randint(10, 20)
        width = random.randint(10, 30)
        question = f"Find the area of a rectangle that is {length} units long and {width} units wide."
        answer = str(length * width)
        
    elif q_type == 8:
        # Perimeter of a rectangle
        length = random.randint(10, 99)
        width = random.randint(10, 99)
        question = f"What is the perimeter of a rectangle with a length of {length} units and a width of {width} units?"
        answer = str(2 * (length + width))
        
    elif q_type == 9:
        # Measurement: comparing lengths using pencils
        pencil_length = random.randint(2, 20)
        multiplier = random.randint(2, 20)
        ruler_length = pencil_length * multiplier
        question = (f"If a pencil is {pencil_length} centimeters long and a ruler is {ruler_length} centimeters long, "
                    "how many pencils laid end-to-end equal the length of the ruler?")
        answer = str(multiplier)
        
    elif q_type == 10:
        # Division with remainder
        friends = random.randint(3, 10)
        candies = random.randint(friends, 50)
        quotient = candies // friends
        remainder = candies % friends
        question = (f"There are {candies} candies shared equally among {friends} friends. "
                    "How many are left over?")
        answer = str(remainder)
        
    elif q_type == 11:
        # Multiplication: Repeated addition (rows of corn)
        rows = random.randint(2, 20)
        plants_per_row = random.randint(2, 50)
        question = (f"A farmer plants {rows} rows of corn with {plants_per_row} corn plants in each row. "
                    "How many corn plants are there altogether?")
        answer = str(rows * plants_per_row)
        
    elif q_type == 12:
        # Subtraction word problem (stickers)
        total_stickers = random.randint(10, 99)
        given_stickers = random.randint(1, total_stickers - 1)
        question = f"Samantha has {total_stickers} stickers. She gives {given_stickers} stickers to her friend. How many stickers does she have left?"
        answer = str(total_stickers - given_stickers)
        
    elif q_type == 13:
        # Fraction addition: Add two fractions and simplify
        denom1 = random.choice([2, 3, 4, 5, 6, 8, 10])
        denom2 = random.choice([2, 3, 4, 5, 6, 8, 10])
        num1 = random.randint(1, denom1 - 1)
        num2 = random.randint(1, denom2 - 1)
        frac_sum = Fraction(num1, denom1) + Fraction(num2, denom2)
        question = f"Add the fractions {num1}/{denom1} and {num2}/{denom2}. Provide your answer in simplest form."
        answer = f"{frac_sum.numerator}/{frac_sum.denominator}"
        
    elif q_type == 14:
        # Time problem: Calculating end time
        duration = random.randint(30, 120)  # in minutes
        start_time = datetime.datetime(2024, 1, 1, 8, 15)  # arbitrary date with time 8:15 AM
        end_time = start_time + datetime.timedelta(minutes=duration)
        # Format the end time in h:mm format (removing any leading zero from the hour)
        end_time_str = end_time.strftime("%I:%M").lstrip("0")
        question = f"School starts at 8:15. If the first class lasts {duration} minutes, what time does the class end?"
        answer = end_time_str
        
    elif q_type == 15:
        # Money problem: Cost of notebooks
        price = round(random.uniform(1.00, 9.99), 2)
        quantity = random.randint(2, 20)
        total_cost = price * quantity
        question = f"One notebook costs ${price:.2f}. If you buy {quantity} notebooks, how much do you spend in total?"
        answer = str(total_cost)
        
    else:
        question = "Error: No question generated."
        answer = "N/A"
    
    return question, answer

def load_config():
    with open("/root/math.yaml", "r") as file:
        return yaml.safe_load(file)

def save_config(math_data):
    with open("/root/math.yaml", "w") as file:
        yaml.safe_dump(math_data, file)

def create_math_yaml(filename='/root/math.yaml'):
    if not os.path.exists(filename):
        data = {
            'accumulated': 0.0,
            'correct': 1.0,
            'incorrect': 0.2,
            'current': {
                'question': '',
                'answer': '',
                'type': None
            },
            'questions': []
        }
        with open(filename, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

def regenerate_yaml():
    # Load the existing configuration
    math_data = load_config()
    
    # Reset accumulated points to zero
    math_data['accumulated'] = 0.0
    
    # Regenerate a new set of 50 random question types (1 to 15)
    amount_questions = 50
    math_data['questions'] = [random.randint(1, 15) for _ in range(amount_questions)]
    
    # Select a new current question from the regenerated list
    if math_data['questions']:
        random_index = random.randint(0, len(math_data['questions']) - 1)
        q_type = math_data['questions'][random_index]
        question, answer = generate_math_question(q_type)
        math_data['current']['question'] = question
        math_data['current']['answer'] = answer
        math_data['current']['type'] = q_type
    else:
        # Handle the case where there are no questions
        math_data['current']['question'] = "No questions available."
        math_data['current']['answer'] = "N/A"
        math_data['current']['type'] = None
    
    # Save the updated configuration back to the YAML file
    save_config(math_data)
    
    # Provide feedback to the user with reset points and the new question
    return f"Accumulated points: {math_data['accumulated']}\n{math_data['current']['question']}", gr.update(value="")

def click_submit(your_Answer):
    math_data = load_config()
    correct_answer = math_data['current']['answer']
    question_type = math_data['current']['type']
    
    # Initialize correctness flag
    is_correct = False
    
    # Check if the answer is related to time problem (q_type == 14)
    if question_type == 14:
        # Normalize both answers by stripping whitespace and converting to lowercase
        user_answer = your_Answer.strip().lower()
        correct = correct_answer.strip().lower()
        
        # Direct string comparison for time in "h:mm" format
        is_correct = user_answer == correct
    else:
        # For other types, attempt numeric comparison
        try:
            # Handle answers with dollar signs or fractions
            if question_type in [10, 15, 13]:
                # Remove any non-numeric characters except for decimal points and slashes
                user_answer_clean = re.sub(r'[^\d./]', '', your_Answer)
                correct_answer_clean = re.sub(r'[^\d./]', '', correct_answer)
                
                # For fractions, compare as fractions
                if '/' in correct_answer_clean:
                    user_frac = Fraction(user_answer_clean)
                    correct_frac = Fraction(correct_answer_clean)
                    is_correct = user_frac == correct_frac
                else:
                    # Compare as float
                    is_correct = float(user_answer_clean) == float(correct_answer_clean)
            else:
                # Direct numeric comparison
                is_correct = float(your_Answer) == float(correct_answer)
        except ValueError:
            # Fallback to string comparison if numeric conversion fails
            is_correct = your_Answer.strip().lower() == correct_answer.strip().lower()
    
    if is_correct:
        math_data['accumulated'] += math_data['correct']
        text = f"Correct!\nAccumulated points: {math_data['accumulated']}"
        try:
            math_data['questions'].remove(math_data['current']['type'])
        except ValueError:
            pass
    else:
        math_data['accumulated'] -= math_data['incorrect']
        text = f"Not Correct! The correct answer was: {correct_answer}\nAccumulated points: {math_data['accumulated']}"
    
    # Generate next question
    if math_data['questions']:
        random_index = random.randint(0, len(math_data['questions']) - 1)
        q_type = math_data['questions'][random_index]
        question, answer = generate_math_question(q_type)
        math_data['current']['question'] = question
        math_data['current']['answer'] = answer
        math_data['current']['type'] = q_type
    else:
        question = "No more questions available."
        answer = "N/A"
    
    save_config(math_data)
    return gr.update(value=f"{text}\n\nNext Question:\n{question}"), gr.update(value="")


def main():
    create_math_yaml()  # Ensure YAML config exists
    config = load_config()
    
    # If 'questions' list is empty, regenerate questions
    if not config['questions']:
        regenerate_yaml()
        config = load_config()

    with gr.Blocks() as demo:
        with gr.Tab("📜 Poem Learning 📜"):
            # Initialize state to store language and poem
            state = gr.State(value={"language": "English", "poem": ""})
            
            with gr.Column():
                # Language selection dropdown
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGE_CODES.keys()),
                    value='English',
                    label='Select Language'
                )
                
                # Poem text input
                poem_input = gr.Textbox(
                    lines=6,
                    placeholder='Enter your poem here with at least 6 lines.',
                    label='Poem'
                )
            
            # Define the update functions inside main to access 'state'
            def update_state_on_language_change(language, current_state):
                """
                Updates the state with the new language while preserving the poem.
                """
                current_poem = current_state.get("poem", "")
                new_state = {"language": language, "poem": current_poem}
                return new_state
            
            def update_state_on_poem_change(new_poem, current_state):
                """
                Updates the state with the new poem while preserving the language.
                """
                current_language = current_state.get("language", "")
                new_state = {"language": current_language, "poem": new_poem}
                return new_state
            
            # Connect the dropdown change to update the state
            language_dropdown.change(
                fn=update_state_on_language_change,
                inputs=[language_dropdown, state],
                outputs=state
            )
            
            # Connect the poem input change to update the state
            poem_input.change(
                fn=update_state_on_poem_change,
                inputs=[poem_input, state],
                outputs=state
            )
            
            with gr.Tab("Practice"):
                with gr.Column():
                    # Line range selectors
                    with gr.Row():
                        start_line = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=1,
                            label='Start Line'
                        )
                        end_line = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=6,
                            label='End Line'
                        )
                    
                    # Generate TTS button and audio output
                    generate_tts_btn = gr.Button("Generate TTS")
                    tts_output = gr.Audio(label="Generated Audio")
                    
                    # Error message output
                    tts_error = gr.Textbox(label="Error Message", visible=False, interactive=False)
                    
                    # Function to handle TTS generation and update state
                    def handle_tts(language, poem, start, end, current_state):
                        lines = poem.strip().split('\n')
                        total_lines = len(lines)

                        audio_path, error = generate_tts(language, poem, (int(start), int(end)))
                        if error:
                            return gr.Audio(value=None), gr.Textbox(value=error, visible=True), gr.Slider(maximum=total_lines), gr.Slider(maximum=total_lines)
                        else:
                            # Update state with current language and poem
                            new_state = {"language": language, "poem": poem}
                            return audio_path, gr.Textbox(value="", visible=False), gr.Slider(maximum=total_lines), gr.Slider(maximum=total_lines)
                    
                    # Connect the button to the TTS function
                    generate_tts_btn.click(
                        fn=handle_tts,
                        inputs=[language_dropdown, poem_input, start_line, end_line, state],
                        outputs=[tts_output, tts_error, start_line, end_line]
                    )
            
            with gr.Tab("Trial"):
                with gr.Column():
                    # Audio input for user's recitation
                    user_audio = gr.Audio(sources=["microphone"], type="filepath", label="Record Your Recitation")
                    
                    # Send button
                    send_btn = gr.Button("Send")
                    
                    # Outputs for transcription and score
                    transcription_output = gr.Textbox(label="Transcription", interactive=False)
                    score_output = gr.Textbox(label="Similarity Score", interactive=False)
                    
                    # Function to handle transcription and scoring
                    def handle_trial(audio, current_state):
                        try:
                            transcription, score = transcribe_and_score(audio, current_state['language'], current_state['poem'])
                            return transcription, score
                        except:
                            return "", 0
                    
                    # Connect the button to the trial function
                    send_btn.click(
                        fn=handle_trial,
                        inputs=[user_audio, state],
                        outputs=[transcription_output, score_output]
                    )
                    
                    # Information about scoring
                    gr.Markdown("### 📊 Score Interpretation")
                    gr.Markdown("""
                    - **0%:** No similarity between your recitation and the poem.
                    - **100%:** Perfect match with the poem.
                    - **Intermediate Values:** Reflect the degree of similarity.
                    """)
            
            # Footer with instructions
            gr.Markdown("""
            ---
            **Instructions:**
            1. Navigate to the **Practice** tab.
            2. Select a language and input a poem with at least six lines.
            3. Choose the range of lines you want to practice.
            4. Click **Generate TTS** to hear the selected lines.
            5. Switch to the **Trial** tab to record your recitation.
            6. Click **Send** to transcribe and score your recitation.
            """)
    
        with gr.Tab("👩‍🔬 Mathematics Learning 🧮"):
            text_markdown = gr.Markdown(f"Points: {config['accumulated']:.1f}\n\n{config['current']['question']}")
            answer_input = gr.Textbox(label="Your answer")
            submit_button = gr.Button("Submit")
            submit_button.click(
                click_submit, 
                inputs=[answer_input], 
                outputs=[text_markdown, answer_input]
            )
            with gr.Accordion("Options", open=False):
                restart_btn = gr.Button("Restart", variant="stop")
                restart_btn.click(regenerate_yaml, outputs=[text_markdown, answer_input])

    # Launch the Gradio app
    demo.launch(server_name="0.0.0.0", server_port=666)
    # demo.launch(server_name="0.0.0.0")

if __name__ == "__main__":
    main()
