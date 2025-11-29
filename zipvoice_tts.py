import logging
from pathlib import Path
import torch
from zipvoice_simplified import load_model, generate_sentence


class ZipVoiceTTS:
    def __init__(self, model_dir: str, lang: str = "vi"):
        """Initialize ZipVoice TTS"""
        self.model_dir = model_dir
        self.lang = lang
        self.model = None
        self.vocoder = None
        self.tokenizer = None
        self.feature_extractor = None
        self.device = None
        self.sampling_rate = None
        
    def load(self):
        """Load model"""
        if self.model is None:
            logging.info("Loading model...")
            (
                self.model,
                self.vocoder,
                self.tokenizer,
                self.feature_extractor,
                self.device,
                self.sampling_rate,
            ) = load_model(self.model_dir, self.lang)
            logging.info("Model loaded successfully!")
        return self
    
    def synthesize(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: str,
        output_path: str,
        speed: float = 1.0,
    ):
        """Synthesize speech from text"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first!")
        
        generate_sentence(
            save_path=output_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            model=self.model,
            vocoder=self.vocoder,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            device=self.device,
            sampling_rate=self.sampling_rate,
            speed=speed,
        )
        logging.info(f"Audio saved to: {output_path}")
        return output_path