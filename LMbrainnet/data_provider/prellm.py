import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class GenPrompt(nn.Module):
    """
    encoder-only (BERT/Roberta/DeBERTa): "first-token"
    decoder-only (GPT2/LLaMA)          : "last-token"
    以及 encoder-decoder (T5/BART)      : "mean-token"

    """   
    def __init__(self,args):  
        super(GenPrompt, self).__init__()
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.chunk_size = args.chunk_size     # 每次处理多少条 prompt（不是 tokens），按显存调整 
        self.max_length=args.max_length
        self.llm_method=args.llm_method
    
        if self.llm_method=='GPT2':
            model_path = '/data/gyun/llm_model/GPT2'
            self.embd_method = "last-token"
        elif self.llm_method=='T5':
            model_path='/data/gyun/llm_model/google-t5/t5-base'
            self.embd_method = "mean-token"
        elif self.llm_method=='BERT':
            model_path='/data/gyun/llm_model/google-bert/bert-base-uncased'
            self.embd_method = "first-token"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False


        # pad 兜底（GPT2/LLaMA 没有 pad）
        if self.tokenizer.pad_token_id is None:
            # 优先用 eos，退化到 unk
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", self.tokenizer.unk_token)
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model_type = getattr(self.model.config, "model_type", "")
        self.is_encdec = bool(getattr(self.model.config, "is_encoder_decoder", False))


    def _pool(self, hidden, attn_mask):
        # hidden: [bsz, T, H], attn_mask: [bsz, T]
        if self.embd_method in ("cls-token", "first-token"):
            return hidden[:, 0, :]
        elif self.embd_method == "last-token":
            idx = attn_mask.sum(dim=1) - 1
            rows = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[rows, idx]
        elif self.embd_method == "mean-token":
            mask = attn_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            raise ValueError(f"Unknown emb_method: {self.emb_method}")


    def _forward_hidden(self, input_ids, attention_mask):
        """
        对 encoder-decoder: 默认拿 encoder 输出作为表征；
        对其余：拿模型输出的 last_hidden_state。
        """
        if self.is_encdec:
            # 直接调 encoder 得到 encoder hidden states，更贴合“语义嵌入”
            encoder = self.model.get_encoder() if hasattr(self.model, "get_encoder") else None
            if encoder is None:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
            else:
                hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
        return hidden

    def generate_embeddings(self, p):
        
        N, B = len(p), len(p[0])
        
        prompt = [p[n][b] for b in range(B) for n in range(N)]
        outs = []

        for s in range(0, B * N, self.chunk_size):
            texts = prompt[s: s + self.chunk_size]
            enc = self.tokenizer(
                texts,
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(self.device, non_blocking=True)
            attn_mask = enc["attention_mask"].to(self.device, non_blocking=True)

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = self._forward_hidden(input_ids, attn_mask)  # [bsz, T, H]
            emb = self._pool(hidden, attn_mask)  # [bsz, H]
            outs.append(emb)

            del input_ids, attn_mask, hidden, emb

        outs = torch.cat(outs, dim=0)        # [B*N, H]
        return outs.view(B, N, -1).contiguous()
