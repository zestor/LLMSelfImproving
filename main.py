#!/usr/bin/env python3

"""
Complete Python script that:
1) Uses local Phi-4 model (quantized, run on CPU via llama.cpp)
2) Accesses Firecrawl & Perplexity AI to gather external data
3) Can optionally call OpenAI GPT-4o or GPT-o1 for external "expert" help
4) Implements MCTS for self-improvement with a local generator + discriminator
5) Fine-tunes (PEFT) the local Phi-4 model on newly generated data

Requirements:
    pip install llama-cpp-python peft transformers torch firecrawl-py perplexityai openai python-dotenv

Environment Variables for external services:
    FIRECRAWL_API_KEY
    PERPLEXITY_API_KEY
    OPENAI_API_KEY

Local files needed:
    /path/to/phi-4-q4.gguf (for MCTS generator / discriminator)
    Some unquantized or partially supported base model "microsoft/phi-4" for PEFT
"""

import os
import re
import json
import torch
import requests
from dotenv import load_dotenv

# Load environment variables from .env if it exists
# We need FIRECRAWL_API_KEY, PERPLEXITY_API_KEY, OPENAI_API_KEY
load_dotenv()

###############################################################################
# 1) llama.cpp local usage for the Phi-4 generator & discriminator
###############################################################################
from llama_cpp import Llama  # for local CPU inference

# 2) Firecrawl Python SDK for scraping/crawling
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False

# 3) Perplexity AI Python usage
try:
    from perplexityai import Perplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

# 4) Official openai python library for GPT-4o or GPT-o1 calls
import openai

# 5) Transformers & PEFT for fine-tuning
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, PeftModel


###############################################################################
# Configuration & Utility
###############################################################################
PHI4_QUANTIZED_PATH = os.environ.get("PHI4_QUANTIZED_PATH", "./phi-4-q4.gguf")
UNQUANTIZED_MODEL_ID = os.environ.get("UNQUANTIZED_MODEL_ID", "microsoft/phi-4")
TOKENIZER_ID         = os.environ.get("TOKENIZER_ID", "microsoft/phi-4")

FIRECRAWL_API_KEY   = os.environ.get("FIRECRAWL_API_KEY", "")
PERPLEXITY_API_KEY  = os.environ.get("PERPLEXITY_API_KEY", "")
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")

# For GPT-4o or GPT-o1 calls
openai.api_key = OPENAI_API_KEY

# Basic checks or usage flags
if FIRECRAWL_API_KEY == "":
    print("[INFO] No FIRECRAWL_API_KEY found. Firecrawl features will fail unless you supply one.")
if PERPLEXITY_API_KEY == "":
    print("[INFO] No PERPLEXITY_API_KEY found. Perplexity calls will fail unless you supply one.")
if OPENAI_API_KEY == "":
    print("[INFO] No OPENAI_API_KEY found. GPT-4o or GPT-o1 usage will fail unless you supply one.")


###############################################################################
# MCTS Generator (local Phi-4)
###############################################################################
class MCTSGenerator:
    """
    Uses a local llama.cpp instance with a quantized Phi-4 model to generate next steps.
    """
    def __init__(self, model_path: str, n_threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Quantized Phi-4 model at {model_path} not found.")
        print(f"[MCTSGenerator] Loading local Phi-4 from {model_path} with {n_threads} threads...")
        self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=n_threads)

    def generate_step(self, partial_reasoning: str) -> str:
        """
        Produce next-step text from the partial reasoning chain.
        """
        prompt = (
            f"Given the reasoning so far:\n\n{partial_reasoning}\n\n"
            f"Continue the reasoning with a single next step:\n"
        )
        result = self.model(
            prompt=prompt,
            max_tokens=64,
            stop=["\n"]
        )
        text_out = result["choices"][0]["text"].strip()
        return text_out


###############################################################################
# MCTS Discriminator (local Phi-4)
###############################################################################
class MCTSDiscriminator:
    """
    Uses a second local llama.cpp instance with the same or separate quantized
    Phi-4 model. Evaluates correctness by returning a numeric score 1-10.
    Optionally merges LoRA adapter if you have a fine-tuned adapter.
    """
    def __init__(self, model_path: str, n_threads: int = 4, lora_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Quantized Phi-4 model at {model_path} not found.")
        print(f"[MCTSDiscriminator] Loading local Phi-4 from {model_path} with {n_threads} threads...")
        self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=n_threads)
        # If llama-cpp-python supported a dynamic LoRA load for phi-4, do it here (feature may not exist yet).
        # For demonstration, we skip that step.

    def score_trajectory(self, trajectory: str) -> float:
        """
        Score the correctness from 1..10. Implementation: prompt the local model
        to produce a single integer (1..10).
        """
        prompt = (
            "Evaluate correctness of the following reasoning path from 1 to 10:\n\n"
            f"{trajectory}\n\n"
            "Score:"
        )
        result = self.model(prompt=prompt, max_tokens=2, stop=["\n"])
        raw = result["choices"][0]["text"].strip()
        # parse integer
        match = re.search(r"(\d+)", raw)
        if match:
            return float(match.group(1))
        return 5.0


###############################################################################
# Monte Carlo Tree Search
###############################################################################
class MCTS:
    """
    Simple MCTS for demonstration: one child expansion per iteration.
    """
    def __init__(self, generator: MCTSGenerator, discriminator: MCTSDiscriminator):
        self.generator = generator
        self.discriminator = discriminator

    def search(self, root_state: str, iterations: int = 5) -> dict:
        """
        MCTS starting from root_state for # iterations.
        Each iteration picks leaf, expands once, scores with discriminator, backprop.
        """
        root = {
            "state": root_state,
            "children": [],
            "parent": None,
            "visits": 0,
            "value": 0.0
        }

        for _ in range(iterations):
            leaf = self._select(root)
            reward = self._simulate(leaf)
            self._backpropagate(leaf, reward)

        return root

    def _select(self, node: dict) -> dict:
        if not node["children"]:
            return node
        best_child = None
        best_ucb = float("-inf")
        c = 2.0
        for child in node["children"]:
            Q = child["value"]
            N = child["visits"] + 1e-6
            total_visits = node["visits"] + 1
            # Approx UCB
            import math
            ucb_value = (Q / N) + c * math.sqrt(math.log(total_visits) / N)
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_child = child
        return best_child

    def _simulate(self, node: dict) -> float:
        # generate a single next step
        next_step = self.generator.generate_step(node["state"])
        new_state = node["state"] + "\n" + next_step
        # attach a new child
        child_node = {
            "state": new_state,
            "children": [],
            "parent": node,
            "visits": 0,
            "value": 0.0
        }
        node["children"].append(child_node)
        # get a reward
        reward = self.discriminator.score_trajectory(new_state)
        return reward

    def _backpropagate(self, leaf: dict, reward: float):
        while leaf is not None:
            leaf["visits"] += 1
            leaf["value"] += reward
            leaf = leaf["parent"]


###############################################################################
# Fine-Tune (PEFT) local Phi-4
###############################################################################
def fine_tune_peft_adapters(
    base_model_id: str,
    tokenizer_id: str,
    new_data: str,
    output_dir: str,
    num_epochs: int = 2
):
    """
    Fine-tunes unquantized Phi-4 with LoRA. This code uses Hugging Face + PEFT.
    Note: Transformers might not fully support "gguf" quant. Typically you must
    load an unquantized or partially quantized model for training. Then re-quant.
    """
    print(f"[PEFT] Loading {base_model_id} for LoRA fine-tuning.")
    model = LlamaForCausalLM.from_pretrained(base_model_id)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )
    model = PeftModel(model, lora_config)

    inputs = tokenizer([new_data], return_tensors="pt", padding=True, truncation=True)
    labels = inputs["input_ids"].clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"[PEFT] Epoch {epoch+1}/{num_epochs} Loss={loss.item():.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"[PEFT] LoRA adapters saved to {output_dir}")


###############################################################################
# 2) Firecrawl usage for referencing external data
###############################################################################
def scrape_with_firecrawl(url: str) -> str:
    """
    Uses Firecrawl to scrape or crawl data from a URL, returning Markdown.
    Requires `firecrawl-py` and a valid `FIRECRAWL_API_KEY`.
    """
    if not FIRECRAWL_AVAILABLE:
        print("[WARN] Firecrawl not installed. Returning empty string.")
        return ""
    if not FIRECRAWL_API_KEY:
        print("[WARN] No Firecrawl API key set. Returning empty string.")
        return ""

    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    print(f"[Firecrawl] Scraping {url} for markdown content.")
    response = app.scrape_url(url, params={"formats": ["markdown"]})
    if response.get("data"):
        return response["data"].get("markdown", "")
    else:
        return ""


###############################################################################
# 3) Perplexity AI usage
###############################################################################
def ask_perplexity(query: str) -> dict:
    """
    Use the perplexityai library to perform a search or answer query.
    Returns the entire dictionary from Perplexity. Typically: {answer, sources, ...}
    Requires `perplexityai` installed and PERPLEXITY_API_KEY set.
    """
    if not PERPLEXITY_AVAILABLE:
        print("[WARN] perplexityai not installed. Returning empty.")
        return {}
    if not PERPLEXITY_API_KEY:
        print("[WARN] No Perplexity API key. Returning empty.")
        return {}

    p = Perplexity(api_key=PERPLEXITY_API_KEY)
    result_list = p.generate_answer(query)
    # perplexity returns a generator; let's return the final chunk
    last_chunk = {}
    for chunk in result_list:
        last_chunk = chunk
    return last_chunk


###############################################################################
# 4) GPT-4o or GPT-o1 usage
###############################################################################
def ask_openai_expert(question: str, model_name: str = "gpt-4o") -> str:
    """
    Optionally call a more capable remote model from OpenAI (e.g., GPT-4o or o1).
    """
    if not OPENAI_API_KEY:
        print("[WARN] No OPENAI_API_KEY. Returning empty string.")
        return ""

    # Since openai python library is used the same for GPT-4 or GPT-4o
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert code and logic assistant."},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] OpenAI call failed: {e}")
        return ""


###############################################################################
# MAIN: Putting It All Together
###############################################################################
def main():
    # 1) Initialize MCTS with local Phi-4 generator & discriminator
    generator = MCTSGenerator(model_path=PHI4_QUANTIZED_PATH, n_threads=4)
    discriminator = MCTSDiscriminator(model_path=PHI4_QUANTIZED_PATH, n_threads=4)

    # 2) Possibly gather external knowledge with Firecrawl & Perplexity
    # Example usage:
    external_data_md = scrape_with_firecrawl("https://firecrawl.dev")
    if external_data_md:
        print("[INFO] Received external Firecrawl data (markdown) length:", len(external_data_md))

    perplexity_res = ask_perplexity("What is the meaning of life?")
    if perplexity_res:
        print("[INFO] Perplexity response excerpt:", perplexity_res.get("answer"))

    # Optional: ask GPT-4o or GPT-o1
    openai_ans = ask_openai_expert("Provide a short, advanced insight about self-improving MCTS in AI", model_name="gpt-4o")
    if openai_ans:
        print("[INFO] GPT-4o answer:\n", openai_ans)

    # 3) Run MCTS on a sample problem
    problem = "Solve x^2 + 5x + 6 = 0. Provide reasoning steps."
    print(f"\n[MCTS] Running on problem: {problem}")
    mcts_system = MCTS(generator, discriminator)
    root_node = mcts_system.search(problem, iterations=3)
    print("[MCTS] Root node after search:", root_node)

    # 4) Gather top validated solutions from MCTS children
    validated_solutions = []
    for child in root_node["children"]:
        score = child["value"] / max(child["visits"], 1e-6)
        if score >= 7.0:
            validated_solutions.append(child["state"])

    # 5) If we have new validated solutions, we can fine-tune using PEFT
    if validated_solutions:
        print("[INFO] Fine-tuning on validated solutions.")
        training_text = "\n\n".join(validated_solutions)
        output_lora_path = "./lora_phi4"
        fine_tune_peft_adapters(
            base_model_id=UNQUANTIZED_MODEL_ID,
            tokenizer_id=TOKENIZER_ID,
            new_data=training_text,
            output_dir=output_lora_path,
            num_epochs=1
        )
    else:
        print("[INFO] No high-scoring solutions from MCTS to fine-tune on.")


if __name__ == "__main__":
    main()
