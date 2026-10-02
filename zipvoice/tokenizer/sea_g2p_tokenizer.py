# Copyright      2024-2026  Xiaomi Corp. & Community
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from sea_g2p import SEAPipeline
except ImportError:
    SEAPipeline = None

from zipvoice.tokenizer.tokenizer import Tokenizer


class SEATokenizer(Tokenizer):
    """Tokenizer backed by SEA-G2P for Southeast Asian languages
    (Vietnamese, Thai, Indonesian) with English code-switching support.
    """

    def __init__(
        self,
        token_file: Optional[Union[str, Path]] = None,
        lang: str = "vi",
    ):
        """
        Args:
            token_file: Path to the file that maps tokens to ids ({token}\\t{id} per line).
            lang: Language code ('vi', 'th', 'id').
        """
        if SEAPipeline is None:
            raise RuntimeError(
                "sea-g2p is not installed. Please install it with:\n"
                "pip install sea-g2p"
            )

        self.lang = lang
        self.pipeline = SEAPipeline(lang=lang)
        self.has_tokens = False
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        if token_file is None:
            logging.debug(
                "Initialize SEATokenizer without tokens file, "
                "will fail when map to ids."
            )
            return

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.rstrip("\r\n")
                if not line:
                    continue
                info = line.split("\t")
                token, token_id = info[0], int(info[1])
                assert token not in self.token2id, f"Duplicate token: {token}"
                self.token2id[token] = token_id
                self.id2token[token_id] = token

        assert "_" in self.token2id, "Token file must contain padding token '_'"
        self.pad_id = self.token2id["_"]
        self.vocab_size = len(self.token2id)
        self.has_tokens = True

    def g2p(self, text: str) -> List[str]:
        """Convert a single text string into a list of IPA character tokens."""
        if not text:
            return []
        phoneme_str = self.pipeline.run(text)
        return list(phoneme_str)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """Convert a list of text strings into a list of token sequences."""
        if not texts:
            return []
        phoneme_strs = self.pipeline.run(texts)
        if isinstance(phoneme_strs, str):
            phoneme_strs = [phoneme_strs]
        return [list(p) for p in phoneme_strs]

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts directly to list of token ID sequences."""
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def tokens_to_token_ids(
        self,
        tokens_list: List[List[str]],
    ) -> List[List[int]]:
        """Map tokens to token IDs, skipping OOVs."""
        assert self.has_tokens, "Please initialize Tokenizer with a tokens file."
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logging.warning(f"[SEATokenizer] Skip OOV token: '{t}' (repr={repr(t)}) - missing from token_file!")
                    continue
                token_ids.append(self.token2id[t])
            token_ids_list.append(token_ids)
        return token_ids_list
