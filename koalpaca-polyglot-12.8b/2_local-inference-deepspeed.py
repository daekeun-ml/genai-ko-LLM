import torch
import deepspeed
import os
import sys
import logging
from types import SimpleNamespace
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, GPTNeoXLayer

import torch.distributed as dist
import torch.multiprocessing as mp

HF_MODEL_ID = "beomi/KoAlpaca-Polyglot-12.8B"
model_name = HF_MODEL_ID.split("/")[-1].replace('.', '-')
model_tar_dir = f"/home/ec2-user/SageMaker/models/{model_name}"

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
sys.path.append('../utils')
sys.path.append('../templates')

def setup(backend="nccl"):

    if 'WORLD_SIZE' in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        rank = int(os.getenv('RANK', 0))
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        # Environment variables set by mpirun 
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', '0'))
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    else:
        sys.exit("Can't find the evironment variables for local rank")
        
    # initialize the process group: 여러 노드에 있는 여러 프로세스가 동기화되고 통신합니다
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    deepspeed.init_distributed(backend)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
    )
    logging.info(f"Initialized the distributed environment. world_size={world_size}, rank={rank}, local_rank={local_rank}")
        
    config = SimpleNamespace()
    config.world_size = world_size
    config.rank = rank
    config.local_rank = local_rank
    config.device = device
    return config


def get_model(properties=None):
    pass

if __name__ == "__main__":
    
    config = setup(backend="nccl")
    print(config)

    with deepspeed.OnDevice(dtype=torch.float16, device="cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_tar_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    ds_config = {
        "tensor_parallel": {"tp_size": config.world_size},
        "dtype": "fp16",
        "injection_policy": {
            GPTNeoXLayer:('attention.dense', 'mlp.dense_4h_to_h')
        }
    }
    model.eval()
    model = deepspeed.init_inference(model, ds_config)

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    generator = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, device=config.local_rank
    )

    from inference_lib import Prompter
    prompter = Prompter("kullm")

    params = {
        "do_sample": False,
        "max_new_tokens": 256,
        "return_full_text": True,
        "temperature": 0.01,
        "top_p": 0.9,
        "return_full_text": False,
        "repetition_penalty": 1.1,
        "presence_penalty": None,
        "eos_token_id": 2,
    }

    instruction = "아래 질문에 대답해줘."
    #instruction = ""
    input_text = "아마존 웹서비스(AWS)에 대해 알려줘"
    prompt = prompter.generate_prompt(instruction, input_text)
    payload = {
        "inputs": [prompt,],
        "parameters": params
    }

    text_inputs, params = payload["inputs"], payload["parameters"]

    from time import perf_counter
    import numpy as np
    latencies = []
    num_infers = 10
    for _ in range(num_infers):
        start_time = perf_counter()
        result = generator(text_inputs, **params)
        latency = perf_counter() - start_time
        latencies.append(latency)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Compute run statistics
        time_avg_sec = np.mean(latencies)
        time_std_sec = np.std(latencies)
        time_p95_sec = np.percentile(latencies, 95)
        time_p99_sec = np.percentile(latencies ,99)
        stats = {"avg_sec": time_avg_sec, "std_sec": time_std_sec, "p95_sec": time_p95_sec, "p99_sec": time_p95_sec}
        print(stats)
        print(result)