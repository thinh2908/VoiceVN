import json
import logging
import gradio as gr
from pathlib import Path
from pydub import AudioSegment
from datetime import datetime
from tqdm import tqdm
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
current_project = None


def load_config():
    """Load config từ file"""
    global config_data, current_profile, current_project
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        current_profile = config_data.get("current_profile", "default")
        current_project = config_data.get("current_project", None)
        
        # Ensure projects key exists
        if "projects" not in config_data:
            config_data["projects"] = {}
            save_config()
        
        return "✓ Config loaded successfully!"
    except FileNotFoundError:
        config_data = {
            "current_profile": "default",
            "current_project": None,
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
            },
            "projects": {}
        }
        save_config()
        return "⚠ Config file not found. Created default config."


def save_config():
    """Save config to file"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)


def get_current_config():
    """Get current profile config"""
    return config_data["profiles"].get(current_profile, config_data["profiles"]["default"])


def get_profile_names():
    """Get list of profile names"""
    return list(config_data["profiles"].keys())


def get_project_names():
    """Get list of project names"""
    return list(config_data["projects"].keys())


# ==================== PROJECT MANAGEMENT ====================

def create_project(project_id, project_name, description, json_file):
    """Create a new project"""
    global current_project
    
    if not project_id:
        return "✗ Project ID cannot be empty!", None, None
    
    if project_id in config_data["projects"]:
        return f"✗ Project '{project_id}' already exists!", None, None
    
    if not json_file:
        return "✗ Please upload a JSON file!", None, None
    
    try:
        # Load and validate JSON
        with open(json_file.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "data" not in data:
            return "✗ Invalid JSON format! Must contain 'data' key.", None, None
        
        # Create project directory
        project_dir = Path(f"./projects/{project_id}")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original JSON
        project_json_path = project_dir / "data.json"
        with open(project_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Create project info
        config_data["projects"][project_id] = {
            "name": project_name or project_id,
            "description": description or "",
            "created_at": datetime.now().isoformat(),
            "json_path": str(project_json_path),
            "output_dir": str(project_dir / "chapters"),
            "total_chapters": len(set(item['id'].split('_')[0] for item in data['data'])),
            "total_segments": len(data['data']),
            "generated_chapters": []
        }
        
        current_project = project_id
        config_data["current_project"] = project_id
        save_config()
        
        return (
            f"✓ Project '{project_id}' created successfully!",
            get_project_info(project_id),
            get_chapters_status(project_id)
        )
    
    except Exception as e:
        return f"✗ Error creating project: {str(e)}", None, None


def switch_project(project_id):
    """Switch to a different project"""
    global current_project
    
    if not project_id:
        return "✗ Please select a project!", None, None
    
    if project_id not in config_data["projects"]:
        return f"✗ Project '{project_id}' not found!", None, None
    
    current_project = project_id
    config_data["current_project"] = project_id
    save_config()
    
    return (
        f"✓ Switched to project: {project_id}",
        get_project_info(project_id),
        get_chapters_status(project_id)
    )


def delete_project(project_id):
    """Delete a project"""
    global current_project
    
    if not project_id:
        return "✗ Please select a project to delete!"
    
    if project_id not in config_data["projects"]:
        return f"✗ Project '{project_id}' not found!"
    
    # Delete project directory
    project_dir = Path(f"./projects/{project_id}")
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)
    
    # Remove from config
    del config_data["projects"][project_id]
    
    # Reset current project if deleted
    if current_project == project_id:
        current_project = None
        config_data["current_project"] = None
    
    save_config()
    
    return f"✓ Project '{project_id}' deleted successfully!"


def get_project_info(project_id):
    """Get project information"""
    if not project_id or project_id not in config_data["projects"]:
        return "No project selected"
    
    proj = config_data["projects"][project_id]
    info = f"""
**Project:** {proj['name']}
**Description:** {proj['description']}
**Created:** {proj['created_at']}
**Total Chapters:** {proj['total_chapters']}
**Total Segments:** {proj['total_segments']}
**Generated Chapters:** {len(proj['generated_chapters'])}/{proj['total_chapters']}
    """
    return info.strip()


def get_chapters_status(project_id):
    """Get chapters generation status"""
    if not project_id or project_id not in config_data["projects"]:
        return "No chapters information"
    
    proj = config_data["projects"][project_id]
    
    # Load JSON to get all chapters
    try:
        with open(proj['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_chapters = sorted(set(item['id'].split('_')[0] for item in data['data']))
        generated = set(proj['generated_chapters'])
        
        status_lines = []
        for chapter in all_chapters:
            status = "✓" if chapter in generated else "✗"
            status_lines.append(f"{status} Chapter {chapter}")
        
        return "\n".join(status_lines)
    
    except Exception as e:
        return f"Error loading chapters: {str(e)}"


def get_pending_chapters(project_id):
    """Get list of chapters that haven't been generated"""
    if not project_id or project_id not in config_data["projects"]:
        return []
    
    proj = config_data["projects"][project_id]
    
    try:
        with open(proj['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_chapters = set(item['id'].split('_')[0] for item in data['data'])
        generated = set(proj['generated_chapters'])
        pending = sorted(list(all_chapters - generated))
        
        return pending
    except:
        return []


# ==================== PROFILE MANAGEMENT ====================

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


def delete_profile(profile_name):
    """Delete a profile"""
    global current_profile
    
    if profile_name == "default":
        return "✗ Cannot delete default profile!"
    
    if profile_name not in config_data["profiles"]:
        return f"✗ Profile '{profile_name}' not found!"
    
    del config_data["profiles"][profile_name]
    
    if current_profile == profile_name:
        current_profile = "default"
        config_data["current_profile"] = "default"
    
    save_config()
    return f"✓ Profile '{profile_name}' deleted successfully!"


# ==================== MODEL MANAGEMENT ====================

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
    """Unload model"""
    global model, vocoder, tokenizer, feature_extractor
    
    model = None
    vocoder = None
    tokenizer = None
    feature_extractor = None
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return "✓ Model unloaded!"


# ==================== GENERATION ====================

def generate_single_audio(text, speed, num_step, guidance_scale):
    """Generate single audio"""
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
        
        return str(output_file), f"✓ Audio generated successfully!"
    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def process_project_chapters(start_chapter, end_chapter, regenerate, progress=gr.Progress()):
    """Process project chapters - with tqdm progress bar"""
    if model is None:
        return None, "✗ Please load model first!", None
    
    if not current_project:
        return None, "✗ Please select a project first!", None
    
    try:
        config = get_current_config()
        proj = config_data["projects"][current_project]
        
        # Load JSON
        with open(proj['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        output_dir = Path(proj['output_dir'])
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
            return None, "✗ No chapters found in range!", None
        
        # Filter out already generated chapters (unless regenerate=True)
        if not regenerate:
            generated_set = set(proj['generated_chapters'])
            sorted_chapters = [
                (ch_id, segs) for ch_id, segs in sorted_chapters
                if ch_id not in generated_set
            ]
            
            if not sorted_chapters:
                return None, "✓ All selected chapters already generated! (Check 'Regenerate' to force)", None
        
        total_chapters = len(sorted_chapters)
        result_files = []
        newly_generated = []
        
        # Create tqdm progress bar for chapters
        chapter_pbar = tqdm(
            sorted_chapters,
            desc="Chapters",
            unit="chapter",
            position=0,
            leave=True
        )
        
        # Process each chapter
        for idx, (chapter_id, segments) in enumerate(chapter_pbar):
            # Update Gradio progress
            progress((idx, total_chapters), desc=f"Processing Chapter {chapter_id}")
            
            # Update tqdm description
            chapter_pbar.set_description(f"Chapter {chapter_id}")
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Starting Chapter {chapter_id} ({idx+1}/{total_chapters})")
            logging.info(f"{'='*60}")
            
            chapter_audio = AudioSegment.empty()
            current_time = 0
            
            # Create tqdm progress bar for segments
            segment_pbar = tqdm(
                segments,
                desc=f"  Segments",
                unit="seg",
                position=1,
                leave=False
            )
            
            # Generate all segments for this chapter
            for seg_idx, segment in enumerate(segment_pbar, 1):
                segment_pbar.set_description(f"  Segment {segment['id']}")
                
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
            
            # Close segment progress bar
            segment_pbar.close()
            
            # Save chapter audio
            chapter_file = output_dir / f"chapter_{chapter_id}.wav"
            chapter_audio.export(str(chapter_file), format="wav")
            result_files.append(str(chapter_file))
            newly_generated.append(chapter_id)
            
            # Update generated chapters list
            proj['generated_chapters'] = list(set(proj['generated_chapters'] + [chapter_id]))
            
            # Save updated JSON with timestamps (incremental)
            output_json = output_dir / "data_updated.json"
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Save config (project info)
            save_config()
            
            # Update chapter progress bar postfix
            chapter_pbar.set_postfix({
                'saved': chapter_file.name,
                'duration': f"{current_time/1000:.1f}s",
                'progress': f"{len(proj['generated_chapters'])}/{proj['total_chapters']}"
            })
            
            logging.info(f"✓ Chapter {chapter_id} saved: {chapter_file}")
            logging.info(f"  Duration: {current_time/1000:.2f}s")
            logging.info(f"  Progress: {len(proj['generated_chapters'])}/{proj['total_chapters']} chapters")
        
        # Close chapter progress bar
        chapter_pbar.close()
        
        # Final status
        return (
            result_files,
            f"✓ Generated {total_chapters} chapters! Total: {len(proj['generated_chapters'])}/{proj['total_chapters']}",
            get_chapters_status(current_project)
        )
    
    except Exception as e:
        logging.error(f"Error in process_project_chapters: {str(e)}")
        return None, f"✗ Error: {str(e)}", get_chapters_status(current_project) if current_project else None

def get_generation_progress():
    """Get current generation progress"""
    if not current_project:
        return "No project selected"
    
    proj = config_data["projects"][current_project]
    generated = len(proj['generated_chapters'])
    total = proj['total_chapters']
    percentage = (generated / total * 100) if total > 0 else 0
    
    return f"Progress: {generated}/{total} chapters ({percentage:.1f}%)"


# Load config at startup
load_config()


# ==================== GRADIO INTERFACE ====================

with gr.Blocks(title="ZipVoice TTS", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ ZipVoice Text-to-Speech with Project Management")
    
    with gr.Row():
        gr.Markdown(f"**Current Profile:** `{current_profile}`")
        gr.Markdown(f"**Current Project:** `{current_project or 'None'}`")
    
    with gr.Tabs():
        # Tab 1: Project Management
        with gr.Tab("📁 Project Management"):
            gr.Markdown("### Create New Project")
            
            with gr.Row():
                with gr.Column():
                    new_proj_id = gr.Textbox(label="Project ID", placeholder="my_audiobook")
                    new_proj_name = gr.Textbox(label="Project Name", placeholder="My Audiobook")
                    new_proj_desc = gr.Textbox(label="Description", lines=2, placeholder="Description...")
                
                with gr.Column():
                    proj_json_file = gr.File(label="Upload JSON File", file_types=[".json"])
            
            create_proj_btn = gr.Button("➕ Create Project", variant="primary")
            create_proj_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Select Project")
            
            with gr.Row():
                project_dropdown = gr.Dropdown(
                    choices=get_project_names(),
                    value=current_project,
                    label="Available Projects",
                    interactive=True
                )
                refresh_proj_btn = gr.Button("🔄 Refresh", size="sm")
            
            with gr.Row():
                switch_proj_btn = gr.Button("✓ Switch Project", variant="primary")
                delete_proj_btn = gr.Button("🗑️ Delete Project", variant="stop")
            
            switch_proj_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Current Project Info")
            
            with gr.Row():
                with gr.Column():
                    proj_info_display = gr.Markdown("No project selected")
                with gr.Column():
                    chapters_status_display = gr.Textbox(
                        label="Chapters Status",
                        lines=10,
                        interactive=False
                    )
            
            # Event handlers
            create_proj_btn.click(
                create_project,
                inputs=[new_proj_id, new_proj_name, new_proj_desc, proj_json_file],
                outputs=[create_proj_status, proj_info_display, chapters_status_display]
            )
            
            switch_proj_btn.click(
                switch_project,
                inputs=[project_dropdown],
                outputs=[switch_proj_status, proj_info_display, chapters_status_display]
            )
            
            delete_proj_btn.click(
                delete_project,
                inputs=[project_dropdown],
                outputs=[switch_proj_status]
            )
            
            refresh_proj_btn.click(
                lambda: gr.update(choices=get_project_names(), value=current_project),
                outputs=[project_dropdown]
            )
        
        # Tab 2: Profile Management
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
            
            profile_info_display = gr.Textbox(label="Profile Info", interactive=False, lines=2)
            
            switch_btn = gr.Button("✓ Switch to Selected Profile", variant="primary")
            switch_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Current Profile Details")
            
            with gr.Row():
                with gr.Column():
                    profile_name = gr.Textbox(label="Profile Name", value=get_current_config().get("name", ""))
                    profile_desc = gr.Textbox(label="Description", value=get_current_config().get("description", ""), lines=2)
                    model_dir_input = gr.Textbox(label="Model Directory", value=get_current_config().get("model_dir", ""))
                    prompt_wav_input = gr.Textbox(label="Prompt Audio File", value=get_current_config().get("prompt_wav", ""))
                
                with gr.Column():
                    prompt_text_input = gr.Textbox(label="Reference Text", value=get_current_config().get("prompt_text", ""), lines=5)
                    output_dir_input = gr.Textbox(label="Output Directory", value=get_current_config().get("output_dir", ""))
            
            with gr.Row():
                speed_input = gr.Slider(0.5, 2.0, get_current_config().get("default_speed", 1.0), step=0.1, label="Speed")
                num_step_input = gr.Slider(4, 32, get_current_config().get("default_num_step", 16), step=1, label="Num Steps")
                guidance_scale_input = gr.Slider(0.0, 5.0, get_current_config().get("default_guidance_scale", 1.0), step=0.1, label="Guidance Scale")
            
            with gr.Row():
                update_profile_btn = gr.Button("💾 Update Current Profile", variant="primary")
                delete_profile_btn = gr.Button("🗑️ Delete Current Profile", variant="stop")
            
            update_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### Create New Profile")
            
            with gr.Row():
                new_profile_id = gr.Textbox(label="Profile ID", placeholder="profile_id")
                new_profile_name = gr.Textbox(label="Display Name", placeholder="My Voice Profile")
            
            new_profile_desc = gr.Textbox(label="Description", placeholder="Description...", lines=2)
            
            create_profile_btn = gr.Button("➕ Create New Profile", variant="secondary")
            create_status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            def update_profile_info(profile_name):
                profile = config_data["profiles"].get(profile_name, {})
                info = f"{profile.get('name', 'N/A')} - {profile.get('description', 'No description')}"
                return info
            
            profile_dropdown.change(update_profile_info, inputs=[profile_dropdown], outputs=[profile_info_display])
            refresh_btn.click(lambda: gr.update(choices=get_profile_names(), value=current_profile), outputs=[profile_dropdown])
            
            switch_btn.click(
                switch_profile,
                inputs=[profile_dropdown],
                outputs=[switch_status, profile_name, profile_desc, model_dir_input, prompt_wav_input, 
                        prompt_text_input, output_dir_input, speed_input, num_step_input, guidance_scale_input]
            )
            
            update_profile_btn.click(
                update_current_profile,
                inputs=[profile_name, profile_desc, model_dir_input, prompt_wav_input, prompt_text_input, 
                       output_dir_input, speed_input, num_step_input, guidance_scale_input],
                outputs=[update_status]
            )
            
            delete_profile_btn.click(lambda: delete_profile(current_profile), outputs=[update_status])
            
            create_profile_btn.click(
                create_new_profile,
                inputs=[new_profile_id, new_profile_name, new_profile_desc, model_dir_input, prompt_wav_input, 
                       prompt_text_input, output_dir_input, speed_input, num_step_input, guidance_scale_input],
                outputs=[create_status]
            )
        
        # Tab 3: Model Management
        with gr.Tab("🤖 Model Management"):
            gr.Markdown("### Load/Unload Model")
            gr.Markdown(f"*Using profile: **{current_profile}***")
            
            with gr.Row():
                load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
                unload_model_btn = gr.Button("🗑️ Unload Model", size="lg")
            
            model_status = gr.Textbox(label="Model Status", interactive=False)
            
            load_model_btn.click(load_model_func, outputs=[model_status])
            unload_model_btn.click(unload_model_func, outputs=[model_status])
        
        # Tab 4: Single Text Generation
        with gr.Tab("🎵 Generate Single Audio"):
            gr.Markdown("### Generate audio from text")
            gr.Markdown(f"*Using profile: **{current_profile}***")
            
            text_input = gr.Textbox(label="Text to synthesize", lines=5, placeholder="Enter text here...")
            
            with gr.Row():
                single_speed = gr.Slider(0.5, 2.0, get_current_config().get("default_speed", 1.0), step=0.1, label="Speed")
                single_num_step = gr.Slider(4, 32, get_current_config().get("default_num_step", 16), step=1, label="Num Steps")
                single_guidance = gr.Slider(0.0, 5.0, get_current_config().get("default_guidance_scale", 1.0), step=0.1, label="Guidance")
            
            generate_btn = gr.Button("🎙️ Generate Audio", variant="primary", size="lg")
            
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            single_status = gr.Textbox(label="Status", interactive=False)
            
            generate_btn.click(
                generate_single_audio,
                inputs=[text_input, single_speed, single_num_step, single_guidance],
                outputs=[audio_output, single_status]
            )
        
        # Tab 5: Generate Project Chapters
        with gr.Tab("📚 Generate Project Chapters"):
            gr.Markdown("### Generate chapters for current project")
            gr.Markdown(f"*Project: **{current_project or 'None selected'}*** | *Profile: **{current_profile}***")
            
            with gr.Row():
                start_chapter_input = gr.Textbox(label="Start Chapter (optional)", placeholder="0001", value="")
                end_chapter_input = gr.Textbox(label="End Chapter (optional)", placeholder="0005", value="")
            
            regenerate_checkbox = gr.Checkbox(label="Regenerate existing chapters", value=False)
            process_btn = gr.Button("🎬 Generate Chapters", variant="primary", size="lg")
            
            chapter_files_output = gr.File(label="Generated Chapter Files", file_count="multiple")
            process_status = gr.Textbox(label="Status", interactive=False)
            process_chapters_status = gr.Textbox(label="Updated Chapters Status", lines=10, interactive=False)
            
            process_btn.click(
                process_project_chapters,
                inputs=[start_chapter_input, end_chapter_input, regenerate_checkbox],
                outputs=[chapter_files_output, process_status, process_chapters_status]
            )

            progress_display = gr.Textbox(
                label="Generation Progress",
                value=get_generation_progress() if current_project else "No project",
                interactive=False
            )

            # Refresh progress button
            refresh_progress_btn = gr.Button("🔄 Refresh Progress")
            refresh_progress_btn.click(
                get_generation_progress,
                outputs=[progress_display]
            )
        
        # Tab 6: Help
        with gr.Tab("❓ Help"):
            gr.Markdown("""
            ## 📖 Usage Guide
            
            ### 1️⃣ Create Project
            - Go to **Project Management** tab
            - Enter Project ID and Name
            - Upload JSON file with format
                  {
                    "data": [
                      {"id": "0001_0001", "text": "...", "prompt": "..."}
                    ]
                  }
    ```
                - Click **Create Project**
                
                ### 2️⃣ Select Profile
                - Go to **Profile Management**
                - Select profile from dropdown
                - Click **Switch to Selected Profile**
                
                ### 3️⃣ Load Model
                - Go to **Model Management**
                - Click **Load Model**
                
                ### 4️⃣ Generate Chapters
                - Go to **Generate Project Chapters**
                - Optionally specify chapter range (e.g., 0001 to 0005)
                - Check **Regenerate** if you want to recreate existing chapters
                - Click **Generate Chapters**
                - System will only generate chapters that haven't been created yet (unless Regenerate is checked)
                
                ### 📊 Track Progress
                - View **Chapters Status** in Project Management tab
                - ✓ = Generated
                - ✗ = Not generated yet
                - Generated chapters are saved incrementally
                - Re-running generation will skip completed chapters
                
                ### 💾 Data Persistence
                - All project data is saved automatically
                - Timestamps (start/end) are preserved in `data_updated.json`
                - Generated chapters are tracked in config
                - Can pause and resume generation anytime
                """)


if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0")

