#!/usr/bin/env python3

import json
import logging
import torch
import torchaudio
from pathlib import Path
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
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



def _finalize_model(model, config, tokenizer, device, model_dir):
    """Finalize model loading: load checkpoint, setup vocoder and feature extractor"""
    model_ckpt = Path(model_dir) / "model.pt"
    
    load_checkpoint(filename=model_ckpt, model=model, strict=True)

    model = model.to(device)
    model.eval()

    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocoder = vocoder.to(device)
    vocoder.eval()

    feature_extractor = VocosFbank()
    sampling_rate = config["feature"]["sampling_rate"]

    return model, vocoder, tokenizer, feature_extractor, device, sampling_rate


def create_tokenizer(tokenizer_type: str = "espeak", token_file = None, lang: str = "vi"):
    """Create tokenizer based on tokenizer_type ('espeak', 'vig2p', 'sea_g2p' / 'sea-g2p', etc.)"""
    tok_type = str(tokenizer_type).lower().replace("-", "_").strip()
    if tok_type == "vig2p":
        from zipvoice.tokenizer.vig2p_tokenizer import ViG2PTokenizer
        return ViG2PTokenizer(token_file=token_file, lang=lang)
    elif tok_type in ("sea_g2p", "seag2p", "sea"):
        from zipvoice.tokenizer.sea_g2p_tokenizer import SEATokenizer
        return SEATokenizer(token_file=token_file, lang=lang)
    elif tok_type == "emilia":
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        return EmiliaTokenizer(token_file=token_file)
    elif tok_type == "libritts":
        from zipvoice.tokenizer.tokenizer import LibriTTSTokenizer
        return LibriTTSTokenizer(token_file=token_file)
    elif tok_type == "simple":
        from zipvoice.tokenizer.tokenizer import SimpleTokenizer
        return SimpleTokenizer(token_file=token_file)
    else:
        from zipvoice.tokenizer.tokenizer import EspeakTokenizer
        return EspeakTokenizer(token_file=token_file, lang=lang)


def load_model(model_dir: str, lang: str = "vi", tokenizer_type: str = "espeak"):
    """Load ZipVoice model va cac components"""
    model_dir = Path(model_dir)
    model_config = model_dir / "model.json"
    token_file = model_dir / "tokens.txt"

    tokenizer = create_tokenizer(tokenizer_type=tokenizer_type, token_file=token_file, lang=lang)

    with open(model_config, "r") as f:
        config = json.load(f)

    model = ZipVoice(
        **config["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device} (tokenizer: {tokenizer_type})")

    return _finalize_model(model, config, tokenizer, device, model_dir)


def load_model_distill(model_dir: str, lang: str = "vi", tokenizer_type: str = "espeak"):
    """Load ZipVoice Distill model va cac components"""
    model_dir = Path(model_dir)
    model_config = model_dir / "model.json"
    token_file = model_dir / "tokens.txt"

    tokenizer = create_tokenizer(tokenizer_type=tokenizer_type, token_file=token_file, lang=lang)

    with open(model_config, "r") as f:
        config = json.load(f)

    model = ZipVoiceDistill(
        **config["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device} (distill mode, tokenizer: {tokenizer_type})")

    return _finalize_model(model, config, tokenizer, device, model_dir)


@torch.inference_mode()
def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer,
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

    tag = "viG2P" if "ViG2PTokenizer" in type(tokenizer).__name__ else ("SEA-G2P" if "SEATokenizer" in type(tokenizer).__name__ else "Phonemes")
    logging.debug(f"[{tag}] Prompt text: {prompt_text}")
    logging.debug(f"[{tag}] Prompt phonemes: {''.join(prompt_tokens_str)}")
    logging.debug(f"[{tag}] Target text: {text}")
    logging.debug(f"[{tag}] Target phonemes: {''.join(tokens_str)}")

    token_duration = (prompt_wav.shape[-1] / sampling_rate) / (
        max(len(prompt_tokens_str), 1) * speed
    )
    max_tokens = int((25 - prompt_duration) / max(token_duration, 1e-4))
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
            batch_wav.append(wav)
        chunked_wavs.extend(batch_wav)

    indexed_chunked_wavs = [(index, wav) for index, wav in zip(chunked_index, chunked_wavs)]
    sequential_chunked_wavs = [wav for _, wav in sorted(indexed_chunked_wavs, key=lambda x: x[0])]
    final_wav = cross_fade_concat(sequential_chunked_wavs, fade_duration=0.1, sample_rate=sampling_rate)
    final_wav = remove_silence(final_wav, sampling_rate, only_edge=(not remove_long_sil), trail_sil=0)

    # Normalize RMS volume consistently across all generations
    cur_rms = torch.sqrt(torch.mean(torch.square(final_wav)))
    norm_target = prompt_rms if (target_rms > 0 and prompt_rms < target_rms) else (target_rms if target_rms > 0 else cur_rms)
    if cur_rms > 1e-5 and norm_target > 0:
        final_wav = final_wav * (norm_target / cur_rms)

    final_wav = final_wav.clamp(-0.99, 0.99)

    # Apply 15ms micro fade-in and fade-out to prevent onset burst and boundary clicks
    fade_len = int(sampling_rate * 0.015)
    if final_wav.shape[-1] > fade_len * 2 and fade_len > 0:
        fade_curve_in = torch.linspace(0.0, 1.0, fade_len, device=final_wav.device)
        fade_curve_out = torch.linspace(1.0, 0.0, fade_len, device=final_wav.device)
        final_wav[..., :fade_len] *= fade_curve_in
        final_wav[..., -fade_len:] *= fade_curve_out

    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)