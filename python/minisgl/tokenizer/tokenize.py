from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    # 将prompt 转换为纯文本，随后进行分词
    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        if not msgs:
            return []

        prompts: List[str] = []


        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            prompts.append(prompt)

        encoded = self.tokenizer(
            prompts,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [torch.tensor(input_ids, dtype=torch.int32) for input_ids in encoded["input_ids"]]
