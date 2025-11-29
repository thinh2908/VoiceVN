import json
import logging
import gradio as gr
from pathlib import Path
from pydub import AudioSegment
import sys
import os
ZIP_VOICE_REPO_PATH = os.getenv("ZIP_VOICE_REPO_PATH")
ZIP_VOICE_APP_PATH= os.getenv("ZIP_VOICE_APP_PATH")

sys.path.append(ZIP_VOICE_REPO_PATH)
sys.path.append(ZIP_VOICE_APP_PATH)


from zipvoice_simplified import load_model, generate_sentence

logging.basicConfig(level=logging.INFO)

CONFIG_FILE = "config.json"

# Global variables
model = None
vocoder = None
tokenizer = None
feature_extractor = None
device = None
sampling_rate = None
config_data = {}
current_profile = "default"


def load_config():
    """Load config từ file"""
    global config_data, current_profile
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        current_profile = config_data.get("current_profile", "default")
        return "✓ Config loaded successfully!"
    except FileNotFoundError:
        # Create default config
        config_data = {
            "current_profile": "default",
            "profiles": {
                "default": {
                    "name": "Default Profile",
                    "description": "Basic Vietnamese TTS",
                    "model_dir": "/content/ZipVoice-Model",
                    "prompt_wav": "/content/1128.MP3",
                    "prompt_text": "YOUR PROMPT TEXT HERE",
                    "output_dir": "/content/output",
                    "default_speed": 1.0,
                    "default_num_step": 16,
                    "default_guidance_scale": 1.0
                }
            }
        }
        save_config()
        return "⚠ Config file not found. Created default config."


def save_config():
    """Save config to file"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    return "✓ Config saved successfully!"


def get_current_config():
    """Get current profile config"""
    return config_data["profiles"].get(current_profile, config_data["profiles"]["default"])


def get_profile_names():
    """Get list of profile names"""
    return list(config_data["profiles"].keys())


def get_profile_info(profile_name):
    """Get profile information for display"""
    profile = config_data["profiles"].get(profile_name, {})
    return (
        profile.get("name", ""),
        profile.get("description", ""),
        profile.get("model_dir", ""),
        profile.get("prompt_wav", ""),
        profile.get("prompt_text", ""),
        profile.get("output_dir", ""),
        profile.get("default_speed", 1.0),
        profile.get("default_num_step", 16),
        profile.get("default_guidance_scale", 1.0)
    )


def switch_profile(profile_name):
    """Switch to a different profile"""
    global current_profile
    if profile_name in config_data["profiles"]:
        current_profile = profile_name
        config_data["current_profile"] = profile_name
        save_config()
        
        profile = config_data["profiles"][profile_name]
        return (
            f"✓ Switched to profile: {profile['name']}",
            profile.get("name", ""),
            profile.get("description", ""),
            profile.get("model_dir", ""),
            profile.get("prompt_wav", ""),
            profile.get("prompt_text", ""),
            profile.get("output_dir", ""),
            profile.get("default_speed", 1.0),
            profile.get("default_num_step", 16),
            profile.get("default_guidance_scale", 1.0)
        )
    return "✗ Profile not found!", "", "", "", "", "", "", 1.0, 16, 1.0


def create_new_profile(profile_id, name, description, model_dir, prompt_wav, 
                       prompt_text, output_dir, speed, num_step, guidance_scale):
    """Create a new profile"""
    if not profile_id:
        return "✗ Profile ID cannot be empty!"
    
    if profile_id in config_data["profiles"]:
        return f"✗ Profile '{profile_id}' already exists!"
    
    config_data["profiles"][profile_id] = {
        "name": name or profile_id,
        "description": description or "",
        "model_dir": model_dir,
        "prompt_wav": prompt_wav,
        "prompt_text": prompt_text,
        "output_dir": output_dir,
        "default_speed": speed,
        "default_num_step": num_step,
        "default_guidance_scale": guidance_scale
    }
    
    save_config()
    return f"✓ Profile '{profile_id}' created successfully!"


def update_current_profile(name, description, model_dir, prompt_wav, 
                          prompt_text, output_dir, speed, num_step, guidance_scale):
    """Update current profile"""
    config_data["profiles"][current_profile].update({
        "name": name,
        "description": description,
        "model_dir": model_dir,
        "prompt_wav": prompt_wav,
        "prompt_text": prompt_text,
        "output_dir": output_dir,
        "default_speed": speed,
        "default_num_step": num_step,
        "default_guidance_scale": guidance_scale
    })
    save_config()
    return f"✓ Profile '{current_profile}' updated successfully!"


def delete_profile(profile_name):
    """Delete a profile"""
    global current_profile
    
    if profile_name == "default":
        return "✗ Cannot delete default profile!"
    
    if profile_name not in config_data["profiles"]:
        return f"✗ Profile '{profile_name}' not found!"
    
    del config_data["profiles"][profile_name]
    
    # Switch to default if deleting current profile
    if current_profile == profile_name:
        current_profile = "default"
        config_data["current_profile"] = "default"
    
    save_config()
    return f"✓ Profile '{profile_name}' deleted successfully!"


def load_model_func():
    """Load ZipVoice model"""
    global model, vocoder, tokenizer, feature_extractor, device, sampling_rate
    
    try:
        if model is not None:
            return "⚠ Model already loaded!"
        
        config = get_current_config()
        logging.info(f"Loading model from {config['model_dir']}...")
        
        (
            model,
            vocoder,
            tokenizer,
            feature_extractor,
            device,
            sampling_rate,
        ) = load_model(config["model_dir"])
        
        return f"✓ Model loaded successfully using profile '{current_profile}'!"
    except Exception as e:
        return f"✗ Error loading model: {str(e)}"


def unload_model_func():
    """Unload model để giải phóng memory"""
    global model, vocoder, tokenizer, feature_extractor
    
    model = None
    vocoder = None
    tokenizer = None
    feature_extractor = None
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return "✓ Model unloaded!"


def generate_single_audio(text, speed, num_step, guidance_scale):
    """Generate một câu audio"""
    if model is None:
        return None, "✗ Please load model first!"
    
    try:
        config = get_current_config()
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "single_output.wav"
        
        generate_sentence(
            save_path=str(output_file),
            prompt_text=config["prompt_text"],
            prompt_wav=config["prompt_wav"],
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            sampling_rate=sampling_rate,
            speed=speed,
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
        
        return str(output_file), f"✓ Audio generated successfully using profile '{current_profile}'!"
    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def process_json_chapters(json_file, start_chapter, end_chapter, progress=gr.Progress()):
    """Process JSON file để tạo chapters"""
    if model is None:
        return None, "✗ Please load model first!"
    
    try:
        config = get_current_config()
        
        # Load JSON
        with open(json_file.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output_dir = Path(config["output_dir"]) / "chapters"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by chapter
        chapters = {}
        for item in data['data']:
            chapter_id = item['id'].split('_')[0]
            chapters.setdefault(chapter_id, []).append(item)
        
        # Filter by range
        sorted_chapters = sorted(chapters.items())
        if start_chapter or end_chapter:
            sorted_chapters = [
                (ch_id, segs) for ch_id, segs in sorted_chapters
                if (not start_chapter or ch_id >= start_chapter) and
                   (not end_chapter or ch_id <= end_chapter)
            ]
        
        if not sorted_chapters:
            return None, "✗ No chapters found in range!"
        
        total_chapters = len(sorted_chapters)
        result_files = []
        
        # Process each chapter
        for idx, (chapter_id, segments) in enumerate(sorted_chapters):
            progress((idx, total_chapters), desc=f"Processing Chapter {chapter_id}")
            
            chapter_audio = AudioSegment.empty()
            current_time = 0
            
            for segment in segments:
                # Generate
                temp_file = output_dir / f"temp_{segment['id']}.wav"
                generate_sentence(
                    save_path=str(temp_file),
                    prompt_text=config["prompt_text"],
                    prompt_wav=config["prompt_wav"],
                    text=segment['text'],
                    model=model,
                    vocoder=vocoder,
                    tokenizer=tokenizer,
                    feature_extractor=feature_extractor,
                    device=device,
                    sampling_rate=sampling_rate,
                    speed=config["default_speed"],
                    num_step=config["default_num_step"],
                    guidance_scale=config["default_guidance_scale"],
                )
                
                # Load and append
                audio = AudioSegment.from_wav(str(temp_file))
                segment['start'] = round(current_time / 1000, 2)
                segment['end'] = round((current_time + len(audio)) / 1000, 2)
                
                chapter_audio += audio
                current_time += len(audio)
                temp_file.unlink()
            
            # Save chapter
            chapter_file = output_dir / f"chapter_{chapter_id}.wav"
            chapter_audio.export(str(chapter_file), format="wav")
            result_files.append(str(chapter_file))
        
        # Save updated JSON
        output_json = output_dir / "data_updated.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return result_files, f"✓ Generated {total_chapters} chapters using profile '{current_profile}'! JSON: {output_json}"
    
    except Exception as e:
        return None, f"✗ Error: {str(e)}"


# Load config at startup
load_config()


# Build Gradio Interface
with gr.Blocks(title="ZipVoice TTS", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ ZipVoice Text-to-Speech")
    gr.Markdown(f"**Current Profile:** `{current_profile}`")
    
    with gr.Tabs():
        # Tab 1: Profile Management
        with gr.Tab("👤 Profile Management"):
            gr.Markdown("### Select Profile")
            
            with gr.Row():
                profile_dropdown = gr.Dropdown(
                    choices=get_profile_names(),
                    value=current_profile,
                    label="Available Profiles",
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            profile_info_display = gr.Textbox(
                label="Profile Info",
                interactive=False,
                lines=2
            )
            
            switch_btn = gr.Button("✓ Switch to Selected Profile", variant="primary")
            switch_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Current Profile Details")
            
            with gr.Row():
                with gr.Column():
                    profile_name = gr.Textbox(
                        label="Profile Name",
                        value=get_current_config().get("name", "")
                    )
                    profile_desc = gr.Textbox(
                        label="Description",
                        value=get_current_config().get("description", ""),
                        lines=2
                    )
                    model_dir_input = gr.Textbox(
                        label="Model Directory",
                        value=get_current_config().get("model_dir", "")
                    )
                    prompt_wav_input = gr.Textbox(
                        label="Prompt Audio File",
                        value=get_current_config().get("prompt_wav", "")
                    )
                
                with gr.Column():
                    prompt_text_input = gr.Textbox(
                        label="Reference Text",
                        value=get_current_config().get("prompt_text", ""),
                        lines=5
                    )
                    output_dir_input = gr.Textbox(
                        label="Output Directory",
                        value=get_current_config().get("output_dir", "")
                    )
            
            with gr.Row():
                speed_input = gr.Slider(
                    0.5, 2.0, get_current_config().get("default_speed", 1.0),
                    step=0.1, label="Speed"
                )
                num_step_input = gr.Slider(
                    4, 32, get_current_config().get("default_num_step", 16),
                    step=1, label="Num Steps"
                )
                guidance_scale_input = gr.Slider(
                    0.0, 5.0, get_current_config().get("default_guidance_scale", 1.0),
                    step=0.1, label="Guidance Scale"
                )
            
            with gr.Row():
                update_profile_btn = gr.Button("💾 Update Current Profile", variant="primary")
                delete_profile_btn = gr.Button("🗑️ Delete Current Profile", variant="stop")
            
            update_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Create New Profile")
            
            with gr.Row():
                new_profile_id = gr.Textbox(
                    label="Profile ID (e.g., 'my_voice')",
                    placeholder="profile_id"
                )
                new_profile_name = gr.Textbox(
                    label="Display Name",
                    placeholder="My Voice Profile"
                )
            
            new_profile_desc = gr.Textbox(
                label="Description",
                placeholder="Description of this profile...",
                lines=2
            )
            
            create_profile_btn = gr.Button("➕ Create New Profile", variant="secondary")
            create_status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            def update_profile_info(profile_name):
                profile = config_data["profiles"].get(profile_name, {})
                info = f"**{profile.get('name', 'N/A')}**\n{profile.get('description', 'No description')}"
                return info
            
            profile_dropdown.change(
                update_profile_info,
                inputs=[profile_dropdown],
                outputs=[profile_info_display]
            )
            
            refresh_btn.click(
                lambda: gr.update(choices=get_profile_names(), value=current_profile),
                outputs=[profile_dropdown]
            )
            
            switch_btn.click(
                switch_profile,
                inputs=[profile_dropdown],
                outputs=[
                    switch_status, profile_name, profile_desc,
                    model_dir_input, prompt_wav_input, prompt_text_input,
                    output_dir_input, speed_input, num_step_input, guidance_scale_input
                ]
            )
            
            update_profile_btn.click(
                update_current_profile,
                inputs=[
                    profile_name, profile_desc, model_dir_input,
                    prompt_wav_input, prompt_text_input, output_dir_input,
                    speed_input, num_step_input, guidance_scale_input
                ],
                outputs=[update_status]
            )
            
            delete_profile_btn.click(
                lambda: delete_profile(current_profile),
                outputs=[update_status]
            )
            
            create_profile_btn.click(
                create_new_profile,
                inputs=[
                    new_profile_id, new_profile_name, new_profile_desc,
                    model_dir_input, prompt_wav_input, prompt_text_input,
                    output_dir_input, speed_input, num_step_input, guidance_scale_input
                ],
                outputs=[create_status]
            )
        
        # Tab 2: Model Management
        with gr.Tab("🤖 Model Management"):
            gr.Markdown("### Load/Unload Model")
            gr.Markdown(f"*Using profile: **{current_profile}***")
            
            with gr.Row():
                load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
                unload_model_btn = gr.Button("🗑️ Unload Model", size="lg")
            
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            load_model_btn.click(load_model_func, outputs=[model_status])
            unload_model_btn.click(unload_model_func, outputs=[model_status])
        
        # Tab 3: Single Text Generation
        with gr.Tab("🎵 Generate Single Audio"):
            gr.Markdown("### Generate audio from text")
            gr.Markdown(f"*Using profile: **{current_profile}***")
            
            text_input = gr.Textbox(
                label="Text to synthesize",
                lines=5,
                placeholder="Enter text here..."
            )
            
            with gr.Row():
                single_speed = gr.Slider(
                    0.5, 2.0, get_current_config().get("default_speed", 1.0),
                    step=0.1, label="Speed"
                )
                single_num_step = gr.Slider(
                    4, 32, get_current_config().get("default_num_step", 16),
                    step=1, label="Num Steps"
                )
                single_guidance = gr.Slider(
                    0.0, 5.0, get_current_config().get("default_guidance_scale", 1.0),
                    step=0.1, label="Guidance"
                )
            
            generate_btn = gr.Button("🎙️ Generate Audio", variant="primary", size="lg")
            
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            single_status = gr.Textbox(label="Status", interactive=False)
            
            generate_btn.click(
                generate_single_audio,
                inputs=[text_input, single_speed, single_num_step, single_guidance],
                outputs=[audio_output, single_status]
            )
        
        # Tab 4: Batch Chapter Generation
        with gr.Tab("📚 Generate Chapters from JSON"):
            gr.Markdown("### Process JSON file to generate chapter audios")
            gr.Markdown(f"*Using profile: **{current_profile}***")
            
            json_file_input = gr.File(label="Upload JSON file", file_types=[".json"])
            
            with gr.Row():
                start_chapter_input = gr.Textbox(
                    label="Start Chapter (optional)",
                    placeholder="0001",
                    value=""
                )
                end_chapter_input = gr.Textbox(
                    label="End Chapter (optional)",
                    placeholder="0005",
                    value=""
                )
            
            process_json_btn = gr.Button("🎬 Generate Chapters", variant="primary", size="lg")
            
            chapter_files_output = gr.File(label="Generated Chapter Files", file_count="multiple")
            json_status = gr.Textbox(label="Status", interactive=False)
            
            process_json_btn.click(
                process_json_chapters,
                inputs=[json_file_input, start_chapter_input, end_chapter_input],
                outputs=[chapter_files_output, json_status]
            )
        
        # Tab 5: Help
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## 📖 Usage Guide
            
            ### 1️⃣ Profile Management
            - **Switch Profile**: Select from dropdown and click "Switch to Selected Profile"
            - **Update Profile**: Modify settings and click "Update Current Profile"
            - **Create New Profile**: 
              1. Fill in Profile ID (e.g., `male_voice`)
              2. Fill in display name and description
              3. Configure settings
              4. Click "Create New Profile"
            - **Delete Profile**: Click "Delete Current Profile" (cannot delete 'default')
            
            ### 2️⃣ Pre-configured Profiles
            - **default**: Basic Vietnamese TTS
            - **male_voice**: Male narrator voice
            - **female_voice**: Female narrator voice
            - **fast_generation**: Quick generation (fewer steps)
            - **high_quality**: Best quality (more steps)
            
            ### 3️⃣ Load Model
            - Go to **Model Management** tab
            - Click **Load Model** (uses current profile's model_dir)
            - Wait for success message
            
            ### 4️⃣ Generate Audio
            - **Single**: Enter text and click "Generate Audio"
            - **Batch**: Upload JSON and click "Generate Chapters"
            
            ### 📝 JSON Format
```json
            {
              "data": [
                {
                  "id": "0001_0001",
                  "text": "Your text here",
                  "prompt": "any_id"
                }
              ]
            }
```
            
            ### 💡 Tips
            - Create profiles for different voices/styles
            - Use fast_generation for testing
            - Use high_quality for final output
            - Profile settings are saved automatically
            """)


if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0")
