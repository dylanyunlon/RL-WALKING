from opencompass.models import HuggingFacewithChatTemplate

# models = [
#     dict(
#         type=HuggingFacewithChatTemplate,
#         abbr='qwen2.5-7b-instruct-hf',
#         path='/workspace/ww/pretrained_models/Qwen25-7B-Instruct',
#         max_out_len=4096,
#         batch_size=8,
#         run_cfg=dict(num_gpus=1),
#     )
# ]

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='Qwen25-7B-Instruct-merge-dou',
        path='/workspace/ww/output/card_qwen-Qwen25-7B-Instruct/merge-doudizhu4-1000000',
        max_out_len=4096,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]