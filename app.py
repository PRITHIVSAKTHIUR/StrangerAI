from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt(message, history, system_prompt=None):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    if system_prompt:
        prompt += f"[SYS] {system_prompt} [/SYS]"
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(
    prompt, history, system_prompt=None, temperature=0.2, max_new_tokens=1024, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history, system_prompt)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output

mychatbot = gr.Chatbot(
    avatar_images=["./user.png", "./botm.png"], bubble_full_width=False, show_label=False, show_copy_button=True, likeable=True,)

demo = gr.ChatInterface(
    fn=generate,
    chatbot=mychatbot,
    title="Hello! I'm a Stranger🗞️.How can I help you today?",
    css="body { background-color: inherit; overflow-x:hidden;}"
                        ":root {--color-accent: transparent !important; --color-accent-soft:transparent !important; --code-background-fill:black !important; --body-text-color:white !important;}"
                        "#component-2 {background:#ffffff1a; display:contents;}"
                        "div#component-0 {    height: auto !important;}"
                        ".gradio-container.gradio-container-4-8-0.svelte-1kyws56.app {max-width: 100% !important;}"
                        "gradio-app {background: linear-gradient(134deg,#00425e 0%,#001a3f 43%,#421438 77%) !important; background-attachment: fixed !important; background-position: top;}"
                        ".panel.svelte-vt1mxs {background: transparent; padding:0;}"
                        ".block.svelte-90oupt {    background: transparent;    border-color: transparent;}"
                        ".bot.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {    background: #ffffff1a;    border-color: transparent;    color: white;}"
                        ".user.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {    background: #ffffff1a;    border-color: transparent;    color: white;    padding: 10px 18px;}"
                        "div.svelte-iyf88w{    background: #cc98d445;    border-color: transparent; border-radius: 25px;}"
                        "textarea.scroll-hide.svelte-1f354aw {    background: transparent; color: #fff !important;}"
                        ".primary.svelte-cmf5ev {   background: transparent;    color: white;}"
                        ".primary.svelte-cmf5ev:hover {   background: transparent;    color: white;}"
                        "button#component-8 {    display: none;    position: absolute;    margin-top: 60px;    border-radius: 25px;}"
                        "div#component-9 {    max-width: fit-content;    margin-left: auto;    margin-right: auto;}"
                        "button#component-10, button#component-11, button#component-12 {    flex: none;    background: #ffffff1a;    border: none;    color: white;    margin-right: auto;    margin-left: auto;    border-radius: 9px;    min-width: fit-content;}"
                        ".share-button.svelte-12dsd9j {    display: none;}"
                        "footer.svelte-mpyp5e {    display: none !important;}"
                        ".message-buttons-bubble.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j { border-color: #31546E;    background: #31546E;}"
                        ".bubble-wrap.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {padding: 0;}"                      
                        ".prose h1 { color: white !important;    font-size: 16px !important;    font-weight: normal !important;    background: #ffffff1a;    padding: 20px;    border-radius: 20px;    width: 90%;    margin-left: auto !important;    margin-right: auto !important;}"
                        ".toast-wrap.svelte-pu0yf1 { display:none !important;}"
                        ".scroll-hide { scrollbar-width: auto !important;}"
                        ".main svelte-1kyws56 {max-width: 800px; align-self: center;}"
                        "div#component-4 {max-width: 650px;    margin-left: auto;    margin-right: auto;}"  
                        "body::-webkit-scrollbar {    display: none;}"
)

demo.queue().launch(show_api=False)