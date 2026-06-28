#!/usr/bin/env python3
"""FIFA RAG: embed the knowledge base with Qwen3-Embedding-0.6B (local) and retrieve top-k for a query."""
import os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

EMB_ID = "Qwen/Qwen3-Embedding-0.6B"
# KB lives next to this script -> resolve relative to the file, not the cwd
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fifa_kb.md")

def load_chunks(path=KB_PATH):
    raw = open(path, encoding="utf-8").read()
    chunks = []
    for block in raw.split("\n\n"):
        lines = [l for l in block.splitlines() if l.strip() and not l.strip().startswith("#")]
        text = " ".join(lines).strip()
        if len(text) > 20:
            chunks.append(text)
    return chunks

class FifaRetriever:
    def __init__(self, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(EMB_ID, padding_side="left")
        self.model = AutoModel.from_pretrained(EMB_ID, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.chunks = load_chunks()
        self.emb = self._embed(self.chunks, is_query=False)

    def _last_token_pool(self, last_hidden, attn_mask):
        # Qwen3-Embedding uses last-token pooling (left-padded -> last column)
        left_pad = attn_mask[:, -1].sum() == attn_mask.shape[0]
        if left_pad:
            return last_hidden[:, -1]
        lengths = attn_mask.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.shape[0]), lengths]

    @torch.no_grad()
    def _embed(self, texts, is_query):
        if is_query:
            instr = "Instruct: Given a football (soccer) question, retrieve relevant FIFA facts\nQuery: "
            texts = [instr + t for t in texts]
        out = []
        for i in range(0, len(texts), 16):
            batch = self.tok(texts[i:i+16], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            hs = self.model(**batch).last_hidden_state
            emb = self._last_token_pool(hs, batch["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, 0)

    def retrieve(self, query, k=4):
        q = self._embed([query], is_query=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in idx]

if __name__ == "__main__":
    r = FifaRetriever()
    print(f"KB chunks: {len(r.chunks)}\n")
    for q in ["Who won the 2022 World Cup?", "What is the offside rule?", "How far is the penalty spot?"]:
        print(f"Q: {q}")
        for s, c in r.retrieve(q, k=2):
            print(f"  [{s:.3f}] {c[:90]}...")
        print()
