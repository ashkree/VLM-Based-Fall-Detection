from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
import torch

SYSTEM_PROMPT = (
"Analyze each input video and determine whether a fall event occurred.\n\n"
"Output Format\n"
"Always respond only in valid JSON following this schema exactly:\n\n"
"{\n"
' "classification": "FALL" | "NO_FALL",\n'
' "confidence": <float between 0.0 and 1.0>,\n'
' "reasoning": <short explanation of why this classification was chosen, up to 600 characters>\n'
' "fall_start": <time in seconds when the fall began, 0 if NO_FALL>\n'
' "fall_end": <time in seconds at moment of impact, 0 if NO_FALL>\n'
"}\n\n"
"Guidelines\n"
"- Falls are scenes where the person experiences rapid and uncontrolled descent resulting in impact.\n"
"- Classify crouching motions, controlled descents, and falling on a bed as NO_FALL.\n"
"- If the classification is NO_FALL, then fall_start and fall_end should be 0.\n"
'- The "confidence" should reflect certainty about the classification.\n'
'- The "reasoning" must summarise key visual cues (e.g., "rapid descent followed by lying posture").\n'
"- Do not include any text outside the JSON.\n"
"- When uncertain or visibility is poor, lower the confidence but still choose the best label.\n"
)

def load(model_name = "Qwen/Qwen2.5-VL-7B-Instruct", use_local=False):

    """
        Loads the model from Hugging Face if use_local = False.
        If use_local is True, loads from local files
    """

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype="float16",
                device_map="auto",
                local_files_only=use_local
            )

    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=use_local
        )

    return model, processor

def prep_message(video_path, system_prompt):

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": "Analyse this video and return only JSON."},
            ],
        },
    ]

def analyze_video(model, processor,  messages, max_new_tokens=128, temperature=0.0):

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts[0]

if __name__ == "__main__":

    import os, json, re
    from pprint import pprint

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    path = "./qwen2.5vl_snapshot"
    
    # check if model is present on the system
    # download if not
    if not os.path.isdir(path):
        snapshot_download(model_name, local_dir=path)

    # load models from local
    model, processor = load(model_name="./qwen2.5vl_snapshot", use_local=True)

    # prep message for inference
    messages = prep_message("./adl-30-cam0.mp4", SYSTEM_PROMPT)
    
    results = analyze_video(model, processor, messages)

    pprint(json.loads(re.sub(r'^```json\n|```$', '', results)))