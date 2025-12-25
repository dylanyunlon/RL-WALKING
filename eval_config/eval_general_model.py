from mmengine.config import read_base
import os.path as osp
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner, VOLCRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    # Datasets Part
    # Knowledge
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_0shot_cot_gen_08c1de import (
        mmlu_pro_datasets,
    )
    from opencompass.configs.datasets.humaneval.humaneval_openai_sample_evals_gen_dcae0e import (
        humaneval_datasets,
    )
    from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen import (
        math_datasets,
    )

    # Summary Groups
    from opencompass.configs.summarizers.groups.mmlu_pro import (
        mmlu_pro_summary_groups,
    )

    # Model List
    from .empty_model import models as empty_model

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################
# datasets list for evaluation
# Only take LCB generation for evaluation
datasets = sum(
    (v for k, v in locals().items() if k.endswith('_datasets')), []
)

#######################################################################
#                       PART 2  Datset Summarizer                     #
#######################################################################

core_summary_groups = [
    {
        'name': 'core_average',
        'subsets': [
            ['mmlu_pro', 'naive_average'],
            ['math_prm800k_500', 'accuracy'],
            ['openai_humaneval', 'humaneval_pass@1'],
        ],
    },
]


summarizer = dict(
    dataset_abbrs=[
        ['core_average', 'naive_average'],
        '',
        'Knowledge',
        ['mmlu_pro', 'naive_average'],
        '',
        'Math Calculation',
        ['math_prm800k_500', 'accuracy'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []
    ),
)

#######################################################################
#                        PART 3  Models  List                         #
#######################################################################

models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

#######################################################################
#                 PART 4  Inference/Evaluation Configuaration         #
#######################################################################

# Local Runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        retry=0,  # Modify if needed
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner, max_num_workers=16, task=dict(type=OpenICLEvalTask)
    ),
)


#######################################################################
#                      PART 5  Utils Configuaration                   #
#######################################################################
# out_dir = '{{$EVAL_OUT_DIR:general}}'
# work_dir = f'./outputs/{out_dir}'