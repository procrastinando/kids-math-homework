import gradio as gr
import random
import yaml
import os

def generate_question():
    operations = ['*', '/', '+', '-']
    num1, num2 = 0, 0
    operation = random.choice(operations)

    if operation in ['+', '-']:
        num1 = random.randint(100, 999)
        num2 = random.randint(1000, 9999)
        if operation == '-':
            num1, num2 = sorted((num1, num2), reverse=True)  # Ensure positive result
        answer = num1 + num2 if operation == '+' else num1 - num2
    elif operation in ['*', '/']:
        num1 = random.randint(10, 99)
        num2 = random.randint(100, 999)
        if operation == '/':
            while True:
                num1, num2 = sorted((random.randint(100, 999), random.randint(10, 99)), reverse=True)
                if num1 % num2 == 0 and num1 // num2 > 1:
                    answer = num1 // num2
                    break
        else:
            answer = num1 * num2

    return f"{num1} {operation} {num2}", answer

def open_config(CONFIG_PATH):
    current_question, correct_answer = generate_question()
    default_config = {
        "correct_increment": 1,
        "wrong_decrement": 0.2,
        "target_points": 100,
        "current_question": current_question,
        "correct_answer": correct_answer,
        "user_points": 0
    }

    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as file:
            yaml.dump(default_config, file)

    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)

def save_config(config):
    with open(CONFIG_PATH, "w") as file:
        yaml.dump(config, file)

def process_input(user_input):
    try:
        user_input = float(user_input)
        if abs(user_input - config['correct_answer']) < 0.1:
            config['user_points'] += config['correct_increment']
            message = f"Correct! Your points: {config['user_points']:.1f}"
        else:
            config['user_points'] -= config['wrong_decrement']
            message = f"Wrong! The correct answer was {config['correct_answer']}. Your points: {config['user_points']:.1f}"

        # Persist updated points and determine next steps
        if config['user_points'] >= config['target_points']:
            save_config(config)
            return "You have finished! Congratulations!", None, gr.update(value="")
        else:
            config['current_question'], config['correct_answer'] = generate_question()
            save_config(config)
            return message, config['current_question'], gr.update(value="")
    except:
        return (
            f"Invalid input. The correct answer was {config['correct_answer']}. \nYour points: {config['user_points']:.1f}",
            generate_question(),
            gr.update(value="")
        )

# Gradio Interface
CONFIG_PATH = "julia.yaml"
config = open_config(CONFIG_PATH)

with gr.Blocks() as demo:
    points_markdown = gr.Markdown(f"Points: {config['user_points']:.1f}")
    current_label = gr.Markdown(config['current_question'])
    answer_input = gr.Textbox(label="Your answer")
    submit_button = gr.Button("Submit")

    submit_button.click(process_input, inputs=[answer_input], outputs=[points_markdown, current_label, answer_input])

demo.launch(server_name="0.0.0.0", server_port=666)