# Self-Improving LLM Reasoning with Local Phi-4

Welcome to the **Self-Improving LLM Reasoning** repository! This project demonstrates how to build a system inspired by rStar-Math that uses:

- A **local quantized Phi-4 model** (via `llama.cpp`) on CPU  
- **MCTS (Monte Carlo Tree Search)** for multi-step reasoning  
- **PEFT (LoRA) fine-tuning** for rapid updates  
- **Firecrawl** and **Perplexity AI** for optional web-based data retrieval  
- **OpenAI GPT-4o** or **GPT-o1** calls for “expert” supplementary checks  

## Concept

1. **Local LLM (Phi-4) for Generation & Discrimination**  
   You run **Phi-4** locally using [llama.cpp](https://github.com/ggerganov/llama.cpp) and the `llama-cpp-python` bindings on CPU only. It acts both as your *generator* (to propose next reasoning steps) and your *discriminator* (to score correctness).

2. **Monte Carlo Tree Search**  
   With each problem, the system expands multiple candidate solution paths. The generator proposes new steps, while the discriminator scores each partial solution, enabling MCTS to converge on promising paths.

3. **PEFT Fine-Tuning**  
   After MCTS identifies valid or high-scoring reasoning paths, the system uses **PEFT (LoRA)** to update the base Phi-4 model’s weights efficiently. Only a small set of adapter parameters are trained, reducing compute requirements.

4. **Supplementing Local LLM with External Knowledge**  
   - **Firecrawl** can scrape or crawl web data to expand the knowledge base.  
   - **Perplexity AI** can provide structured answers and citations for deeper insight.  
   - **OpenAI GPT-4o** or **o1** can be queried as an external “expert,” if needed, to check or refine solutions.

By repeating this cycle—*generate, discriminate, fine-tune*—the system incrementally improves its performance on specific tasks without requiring GPU resources.

## Requirements

You need:

1. A **quantized** Phi-4 model (GGUF format) for local inference on CPU (e.g., `phi-4-q4.gguf`).  
2. An **unquantized** or partially quantized Phi-4 model for fine-tuning, such as the [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) repository .  
3. Python 3.8+ environment.  
4. The following libraries to be installed:

```bash
pip install llama-cpp-python peft transformers torch firecrawl-py perplexityai openai python-dotenv
```

5. (Optional) **API Keys**:  
   - Firecrawl: `FIRECRAWL_API_KEY`  
   - Perplexity: `PERPLEXITY_API_KEY`  
   - OpenAI: `OPENAI_API_KEY` (required only if you want GPT-4o or GPT-o1 calls)

## Repository Structure

```
self_improving_phi4/
├─ README.md            # This README
├─ main.py   # Main code demonstrating the entire flow
└─ .env.example         # Example environment variables file
```

- **self_improving_phi4.py**  
  The complete script containing:
  - MCTS generator & discriminator using local Phi-4  
  - MCTS implementation  
  - Firecrawl & Perplexity integration  
  - GPT-4o / GPT-o1 integration  
  - PEFT LoRA fine-tuning

- **.env.example**  
  An example showing how to specify environment variables:
  ```
  FIRECRAWL_API_KEY=fc-XXXXXXXXXXXXXXXXXXXX
  PERPLEXITY_API_KEY=pplx-XXXXXXXXXXXXXXXXXX
  OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXX
  PHI4_QUANTIZED_PATH=./phi-4-q4.gguf
  UNQUANTIZED_MODEL_ID=microsoft/phi-4
  TOKENIZER_ID=microsoft/phi-4
  ```

## Setup

1. **Clone or download** this repository.  
2. **Download the Quantized Phi-4 model** (`phi-4-q4.gguf`) from [microsoft/phi-4-gguf](https://huggingface.co/microsoft/phi-4-gguf/tree/main) and place it in the repository folder.  
3. **Install Python dependencies**:

   ```bash
   pip install llama-cpp-python peft transformers torch firecrawl-py perplexityai openai python-dotenv
   ```

4. **Make a copy** of `.env.example` to `.env`, and fill in the needed API keys (if using Firecrawl, Perplexity, or OpenAI). Example:

   ```bash
   cp .env.example .env
   ```

5. **Ensure** you have the unquantized Phi-4 model for fine-tuning. By default, we use `"microsoft/phi-4"` from Hugging Face in `UNQUANTIZED_MODEL_ID`.

## Usage

1. **Local Execution**  
   Run the main script:

   ```bash
   python self_improving_phi4.py
   ```

   The script will:  
   - Load your local quantized model (`phi-4-q4.gguf`) for MCTS generation and discrimination.  
   - Attempt MCTS on a sample problem.  
   - Collect “valid” solutions and fine-tune the unquantized base model with LoRA if any solutions pass the threshold.  

2. **Firecrawl & Perplexity**  
   - If `FIRECRAWL_API_KEY` and `PERPLEXITY_API_KEY` are set, the script will demonstrate scraping and retrieving external data.  
   - If you do not have those keys, the script will skip those parts (printing warnings).

3. **OpenAI GPT-4o / GPT-o1**  
   - If `OPENAI_API_KEY` is set, the script will demonstrate calling GPT-4o or GPT-o1 for advanced external checks.  
   - If not, it will gracefully skip them.

4. **PEFT Fine-Tuning**  
   - The code automatically runs a minimal fine-tuning example on the validated solutions found by MCTS.  
   - This produces LoRA adapter weights in `./lora_phi4` by default (configurable in the code).  
   - For repeated cycles, you could incorporate these LoRA weights back into your local quantized model by merging them manually (requires a separate merging tool, as llama-cpp-python does not fully support dynamic LoRA loading for Phi-4 yet).

## Additional Notes

- This example is **CPU-only**, so performance will be slower than GPU-based solutions. Reducing context sizes, tokens, or iteration counts can help with speed.  
- MCTS is kept **simple** for demonstration. Real use cases may require improved heuristics, parallel expansions, or more robust state management.  
- **LoRA merging**: You may need to re-quantize your model or load LoRA via a specialized script to apply the adapters in the local llama.cpp environment.

## FAQ

1. **Why can't I load LoRA adapters directly in llama.cpp for Phi-4?**  
   - As of this writing, official support for loading LoRA adapters into llama.cpp for all models is limited. You may need a manual merge step to produce a new GGUF file.

2. **Do I need Firecrawl or Perplexity for the code to run?**  
   - No. The code runs locally with the Phi-4 model by default. If you have valid keys, it will supplement reasoning with external data.

3. **Is the code production-ready?**  
   - This code provides a working demonstration. For production, ensure you handle error cases, concurrency, model versioning, and compliance with the terms of service for external APIs.

## Reference

-  Phi-4 model from Microsoft: https://huggingface.co/microsoft/phi-4-gguf/tree/main
