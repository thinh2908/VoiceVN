#!/usr/bin/env python3

import json
import logging
import torch
import torchaudio
from pathlib import Path
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)


def load_model(model_dir: str, lang: str = "vi"):
    """Load ZipVoice model và các components"""
    model_dir = Path(model_dir)
    model_ckpt = model_dir / "model.pt"
    model_config = model_dir / "model.json"
    token_file = model_dir / "tokens.txt"

    tokenizer = EspeakTokenizer(token_file=token_file, lang=lang)
    
    with open(model_config, "r") as f:
        config = json.load(f)

    model = ZipVoice(
        **config["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    
    load_checkpoint(filename=model_ckpt, model=model, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocoder = vocoder.to(device)
    vocoder.eval()

    feature_extractor = VocosFbank()
    sampling_rate = config["feature"]["sampling_rate"]

    return model, vocoder, tokenizer, feature_extractor, device, sampling_rate


@torch.inference_mode()
def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EspeakTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    """Generate speech từ text"""
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    prompt_wav = remove_silence(prompt_wav, sampling_rate, only_edge=False, trail_sil=200)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)
    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=sampling_rate).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)

    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (len(prompt_tokens_str) * speed)
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)

    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

    tokens_batches, chunked_index = batchify_tokens(chunked_tokens, max_duration, prompt_duration, token_duration)

    chunked_features = []
    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)
        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full((len(batch_tokens),), prompt_features.size(1), device=device)

        pred_features, pred_features_lens, _, _ = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

        pred_features = pred_features.permute(0, 2, 1) / feat_scale
        chunked_features.append((pred_features, pred_features_lens))

    chunked_wavs = []
    for pred_features, pred_features_lens in chunked_features:
        batch_wav = []
        for i in range(pred_features.size(0)):
            wav = vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]]).squeeze(1).clamp(-1, 1)
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    indexed_chunked_wavs = [(index, wav) for index, wav in zip(chunked_index, chunked_wavs)]
    sequential_chunked_wavs = [wav for _, wav in sorted(indexed_chunked_wavs, key=lambda x: x[0])]
    final_wav = cross_fade_concat(sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate)
    final_wav = remove_silence(final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0)

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)