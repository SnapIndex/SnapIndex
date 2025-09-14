import argparse
import sys
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_prompt(messages: List[dict]) -> List[dict]:
	return messages


def main() -> None:
	parser = argparse.ArgumentParser(description="Minimal terminal chat with Transformers + Qwen")
	parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
	parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
	parser.add_argument("--max_new_tokens", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top_p", type=float, default=0.95)
	args = parser.parse_args()

	device = args.device
	if device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(
		args.model,
		torch_dtype=torch.float32,
		device_map="auto",
	)
	if device == "cpu":
		model = model.to("cpu")

	messages: List[dict] = [
		{"role": "system", "content": "You are a helpful assistant."}
	]

	print(f"Loaded {args.model} on {device}. Type 'exit' to quit.")
	while True:
		try:
			user_input = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break
		if user_input.lower() in {"exit", "quit"}:
			break

		messages.append({"role": "user", "content": user_input})

		# Apply chat template if available
		if hasattr(tokenizer, "apply_chat_template"):
			prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
		else:
			inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

		with torch.no_grad():
			outputs = model.generate(
				inputs.input_ids,
				attention_mask=inputs.attention_mask,
				max_new_tokens=args.max_new_tokens,
				temperature=args.temperature,
				top_p=args.top_p,
				do_sample=True,
			)

		response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
		print(f"Assistant: {response_text}")
		messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		sys.exit(1)
