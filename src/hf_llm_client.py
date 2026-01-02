# src/hf_llm_client.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class HFConfig:
    enabled: bool
    model_dir: str
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float
    lang: str

    @staticmethod
    def from_env() -> "HFConfig":
        return HFConfig(
            enabled=os.getenv("RAG_USE_HF_LLM", "0").strip() == "1",
            model_dir=os.getenv("HF_MODEL_DIR", "").strip(),
            device=os.getenv("HF_DEVICE", "cpu").strip(),
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "180").strip()),
            temperature=float(os.getenv("HF_TEMPERATURE", "0.0").strip()),
            top_p=float(os.getenv("HF_TOP_P", "1.0").strip()),
            lang=os.getenv("HF_LANG", "en").strip(),  # "en" or "ar"
        )



def build_rag_prompt(question: str, context: str) -> str:
    return f"""You are a grounded assistant. Write in English only.

Use ONLY the Context below. If the Context does not contain the answer, say exactly:
Not found in the provided documents.

Rules:
- 2 to 6 bullet points maximum.
- After each bullet, add a citation like (S1) or (S2) based on the context labels.
- Do NOT mention methods (e.g., TF-IDF, BM25) unless they appear in the Context.
- Do NOT repeat yourself.

Context:
{context}

Question:
{question}

Answer:
"""




class HFLocalLLM:
    def __init__(self, cfg: HFConfig) -> None:
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if not self.cfg.model_dir:
            raise ValueError("HF_MODEL_DIR is empty. Set HF_MODEL_DIR to your local model folder path.")

        tok = AutoTokenizer.from_pretrained(self.cfg.model_dir, local_files_only=True)
        # Use float32 on CPU for stability
        dtype = torch.float16 if (self.cfg.device.startswith("cuda")) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_dir,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if self.cfg.device.startswith("cuda") else None,
        )

        if not self.cfg.device.startswith("cuda"):
            model.to(self.cfg.device)

        model.eval()

        self._tokenizer = tok
        self._model = model

    @property
    def ready(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def generate(self, prompt: str) -> str:
        if not self.ready:
            self.load()

        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if not self.cfg.device.startswith("cuda"):
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
    **inputs,
    max_new_tokens=self.cfg.max_new_tokens,
    do_sample=False,                 # مهم جدًا: بدون sampling
    repetition_penalty=1.12,
    no_repeat_ngram_size=4,
    eos_token_id=self._tokenizer.eos_token_id,
    pad_token_id=self._tokenizer.eos_token_id,
)


        text = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # Return only the tail after "Answer:" if present
        marker = "Answer:"
        if marker in text:
            text = text.split(marker, 1)[-1].strip()

        return text
