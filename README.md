# Can Large Language Models Master Complex Card Games?

This repository contains information and code of Can Large Language Models Master Complex Card Games?.

## Introduction

We investigate whether language models can master complex card games. We systematically evaluate the performance of language models on eight carefully selected card games. Specifically, we focus on the following three research questions:

Can LLMs master complex card games? And how much data is required for them to master these games?

Can LLMs simultaneously master multiple games? Do different games mutually enhance each other or do conflicts arise between them?

Can LLMs maintain their general capabilities while mastering complex games?

To answer these questions, we first fine-tune language models on each of the eight games separately to evaluate the extent to which the models can master individual games. Next, we fine-tune the models on a mixture of all the game data to assess their ability to master all the games simultaneously. Finally, we evaluate whether the models' general capabilities decline using MMLU-Pro, Math-500, and HumanEval benchmarks for knowledge question answering, math, and coding skills.

## Installation

To collect game data and evaluate game performance, we rely on three projects: [DouZero](https://github.com/kwai/DouZero), [DanZero](https://github.com/submit-paper/Danzero_plus), and [RLCard](https://github.com/datamllab/rlcard).  For training, we use the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework, and for evaluating general benchmarks, we use the [OpenCompass](https://github.com/open-compass/opencompass) framework. Therefore, it is necessary to install the dependencies for these five projects.

## Generate data

First, use the teacher model to generate interaction data for each game:

```sh
bash gen_data.sh
```

Then, filter and covert interaction data to sft format:

```sh
bash convert_data.sh
```

## Training

Run the following script to train a mixture model on the data from the 8 games:

```sh
bash train.sh
```

Execute the following script to further train the mixture model with general data, in order to mitigate the issue of degraded general capabilities in the hybrid model:

```sh
bash train_ct.sh
```

## Evaluation

### Question 1

Use the following script to evaluate the change in model performance as the data volume increases:

```sh
bash de_ckpt.sh
```

### Question 2

Use the following script to verify the performance of the hybrid model on 8 games:

```sh
bash de_final.sh
```

Use the following script to validate the performance of the API-based models on 8 games:

```sh
bash eval_llm_one_on_all.sh
```

### Question 3

Use the following script to verify the performance of models on three general benchmarks:

```sh
bash eval_general.sh
```

## Acknowledgement

Our project utilizes many open-source projects. We thank the contributors of these projects, including but not limited to: [DouZero](https://github.com/kwai/DouZero), [DanZero](https://github.com/submit-paper/Danzero_plus), [RLCard](https://github.com/datamllab/rlcard), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [OpenCompass](https://github.com/open-compass/opencompass).

## Citation

If you find our work helpful, please kindly cite our paper:

```latex
@article{wang2025can,
  title={Can Large Language Models Master Complex Card Games?},
  author={Wang, Wei and Bie, Fuqing and Chen, Junzhe and Zhang, Dan and Huang, Shiyu and Kharlamov, Evgeny and Tang, Jie},
  journal={arXiv preprint arXiv:2509.01328},
  year={2025}
}
```
