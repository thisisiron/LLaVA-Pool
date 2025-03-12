import os
import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoModelForVision2Seq, AutoModelForImageTextToText
import argparse
import re

class ModelInferenceSystem:
    def __init__(self, model_path):
        """Load the model and processor."""
        print(f"Loading model from {model_path}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        if "qwen" in self.model.config.model_type:
            self.processor.image_processor.max_pixels = 28 * 28 * 1280
        print("Model loaded successfully!")

    def process_image_and_text(self, image, prompt):
        """Process the image and text to generate model output."""
        if image is None:
            return "Please upload an image or provide an image path."
        
        try:
            # Create a message structure in chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # (1) Generate input text in chat format
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # (2) Preprocess vision information (image)
            image_inputs = [image]
            
            # (3) Generate model input through processor (text, image)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to the model's device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generation settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Separate user prompt and model response
            # Since a chat template was used, the response extraction method needs to be changed
            return generated_text.strip()
                
        except Exception as e:
            return f"Error processing image and text: {str(e)}"

    def load_image_from_path(self, image_path):
        """Load an image from an internal server path."""
        if not image_path:
            return None
            
        try:
            if not os.path.exists(image_path):
                return f"Image path does not exist: {image_path}"
            
            image = Image.open(image_path)
            return image
        except Exception as e:
            return f"Error loading image from path: {str(e)}"

def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LLaVA-Pool Image and Text Inference")
        gr.Markdown("Upload an image or provide an image path, enter a prompt, and get a response from the model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Enter the path to your model (local or Hugging Face model ID)",
                    value=""
                )
                load_button = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", value="No model loaded", interactive=False)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Upload Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="pil", label="Upload Image")
                        prompt_upload = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here",
                            value="",
                            lines=3
                        )
                        run_button_upload = gr.Button("Run Inference", variant="primary")
                    with gr.Column(scale=1):
                        with gr.Tabs() as output_tabs_upload:
                            with gr.TabItem("Raw"):
                                output_upload = gr.Textbox(label="Raw Output", lines=10)
                            with gr.TabItem("Markdown"):
                                output_upload_markdown = gr.Markdown(
                                    label="Markdown Rendered Output",
                                )
            
            with gr.TabItem("Image Path"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_path = gr.Textbox(
                            label="Image Path",
                            placeholder="Enter the server path to your image file (e.g., /home/infidea/eon/code/LLaVA-Pool/data/demo_data/COCO_train2014_000000222016.jpg)",
                            value=""
                        )
                        prompt_path = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here",
                            value="",
                            lines=3
                        )
                        run_button_path = gr.Button("Run Inference", variant="primary")
                    with gr.Column(scale=1):
                        with gr.Tabs() as output_tabs_path:
                            with gr.TabItem("Raw"):
                                output_path = gr.Textbox(label="Raw Output", lines=10)
                            with gr.TabItem("Markdown"):
                                output_path_markdown = gr.Markdown(label="Markdown Rendered Output")
                        preview_image = gr.Image(type="pil", label="Image Preview")
            
        # Store model instance as a global variable
        model_instance = {"model": None}
        
        def load_model(model_path):
            try:
                model_instance["model"] = ModelInferenceSystem(model_path)
                return f"Model loaded successfully from {model_path}"
            except Exception as e:
                return f"Error loading model: {str(e)}"
        
        def extract_markdown_content(text):
            """Extract content from a markdown code block."""
            # Markdown code block pattern (```markdown ... ```)
            pattern = r"```markdown\s*([\s\S]*?)\s*```"
            match = re.search(pattern, text)
            
            if match:
                # Extract only the content of the code block
                return match.group(1).strip()

            # Return original if not a markdown code block
            return text

        def run_inference_upload(image, prompt):
            if model_instance["model"] is None:
                return "Please load a model first.", "Please load a model first."

            result = model_instance["model"].process_image_and_text(image, prompt)
            markdown_content = extract_markdown_content(result)

            return result, markdown_content

        def run_inference_path(image_path, prompt):
            if model_instance["model"] is None:
                return "Please load a model first.", "Please load a model first.", None
            
            image = None
            try:
                if not os.path.exists(image_path):
                    return "Image path does not exist: " + image_path, "Image path does not exist: " + image_path, None
                
                image = Image.open(image_path)
            except Exception as e:
                error_msg = f"Error loading image from path: {str(e)}"
                return error_msg, error_msg, None
                
            result = model_instance["model"].process_image_and_text(image, prompt)
            markdown_content = extract_markdown_content(result)
            
            return result, markdown_content, image
        
        def preview_image_from_path(image_path):
            try:
                if not os.path.exists(image_path) or not image_path:
                    return None
                
                image = Image.open(image_path)
                return image
            except Exception as e:
                return None
        
        # Connect events
        load_button.click(load_model, inputs=[model_path], outputs=model_status)
        run_button_upload.click(run_inference_upload, inputs=[image_input, prompt_upload], outputs=[output_upload, output_upload_markdown])
        run_button_path.click(run_inference_path, inputs=[image_path, prompt_path], outputs=[output_path, output_path_markdown, preview_image])
        
        # Automatically update preview when image path is entered
        image_path.change(preview_image_from_path, inputs=[image_path], outputs=preview_image)
        
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-Pool Inference UI")
    parser.add_argument("--share", action="store_true", help="Create a public link to share the app")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        share=args.share
    )