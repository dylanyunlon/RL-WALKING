from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf',
        path='/workspace/ww/pretrained_models/Qwen25-7B-Instruct',
        peft_path='/workspace/ww/pretrained_models/Qwen25-7B-Instruct-peft',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]