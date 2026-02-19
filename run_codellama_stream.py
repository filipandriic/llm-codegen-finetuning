from llama_cpp import Llama

llm = Llama(
    model_path="/Users/filipandric/codellama_finetune/artifacts/tinyllama-3ep/model_finetuned.Q5_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=100,
)

SYSTEM = (
    "You are a helpful programming assistant. "
    "Follow the instructions exactly and keep code consistent with project conventions."
)

def make_inst_prompt(system: str, user: str) -> str:
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user}\n[/INST]\n"

def gen_stream(prompt, max_tokens=1024, temperature=0.7):
    print("AI: ", end="", flush=True)
    try:
        for chunk in llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stop=["</s>", "[INST]", "[/INST]"],
        ):
            text = chunk["choices"][0]["text"]
            print(text, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[STOPPED BY USER]")
    print()

if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        prompt = make_inst_prompt(SYSTEM, user_input)
        gen_stream(prompt)
