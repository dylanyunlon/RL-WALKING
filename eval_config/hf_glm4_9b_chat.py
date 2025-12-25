from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='glm-4-9b-chat-hf',
        path='/workspace/ww/pretrained_models/glm-4-9b-chat',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|endoftext|>', '<|user|>', '<|observation|>'],
    )
]