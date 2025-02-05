# Usage

## Install
```bash
conda create --name openrlhf python=3.10.16
pip install vllm==0.6.5
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
pip install openrlhf
pip install "git+https://github.com/tongyx361/symeval.git"
```

## Training 
```bash
cd OpenRLHF-STILL

## Multi Node
bash dpsk1_5b-4nodes-grpo.sh

## Single Node
bash dpsk1_5b-1node-grpo.sh

```

## Evaluation

```bash
cd OpenRLHF-STILL/evaluation

bash scripts/run_eval.sh

```

## Results


| | MATH | AIME | OMNI | LiveAOPS | Avg. |
| --- | --- | --- | --- | --- | --- |
|Qwen-2.5-Math-7B-Instruct|83.60|16.67|	- | -| - |
|Qwen-2.5-Math-72B-Instruct|85.90|30.00|	- | -| - |
|O1-preview	| 85.50 | 44.60 |	- | -| - |
|STILL-2	| 90.20	| 46.67	| -	| - | -|
|QwQ-32B	| 90.60	| 50.00	| -	| - | -|
| DeepSeek-R1-Distill-Qwen-1.5B | 84.04 | 28.67 | 25.60 | 33.33 | 42.91 |
| STILL-3-1.5B-preview | **85.48** | **39.33** | **33.00** | **39.50** | **49.33** |

