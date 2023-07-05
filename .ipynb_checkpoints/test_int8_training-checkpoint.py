import torch

from mpt import MPTConfig, MPTForCausalLM, load_model

from peft import prepare_model_for_int8_training,  get_peft_model
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import transformers
from typing import Dict, Any
from gradient_checkpointing import apply_gradient_checkpointing
from mpt import MPTBlock

from transformers import (

    CONFIG_MAPPING,

    MODEL_FOR_CAUSAL_LM_MAPPING,

    AutoConfig,

    AutoModelForCausalLM,

    AutoTokenizer,

    HfArgumentParser,

    Trainer,

    TrainingArguments,

    default_data_collator,

    is_torch_tpu_available,

    set_seed,

)
import os
os.environ['WANDB_MODE']='offline'
import argparse
parser = argparse.ArgumentParser(
        description="Produce MPT in 4-bit training"
    )
parser.add_argument("--local_rank", type=int, default=0, help="local rank if using torch.distributed.launch")
parser.add_argument('--model_cache_dir', type=str, required=True, help='Path to model weights.')
parser.add_argument('--cache_dir', type=str, required=True, help='Path to local HF cache.')
# model_save_dir
parser.add_argument('--model_save_dir', type=str, required=True, help='Path to local model dir.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to local dataset.')
from datasets import Dataset, load_dataset
from transformers.utils import logging

logger = logging.get_logger("transformers")

from abc import ABC, abstractmethod

class TrainDataBase(ABC):
    """
    """
    @abstractmethod
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len: int) -> None:
        """
        Args:
            dataset (str): Path to dataset
            val_set_size (int) : Size of validation set
            tokenizer (_type_): Tokenizer
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_set_size = val_set_size
        self.cutoff_len = cutoff_len
        self.train_data = None
        self.val_data = None

    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        """Loads dataset from file and prepares train_data for trainer."""
        pass


class TrainSAD(TrainDataBase):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        data = load_dataset("json", data_files=self.dataset)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(test_size=self.val_set_size, shuffle=True, seed=42)
            self.train_data = train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        else:
            self.train_data = data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = None

    # Auxiliary methods
    def generate_prompt(self, data_point, **kwargs):
        return make_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"]
        )

    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)


def make_prompt(instruction, input_, output=""):
    return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}\n{6}".format(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        "### Instruction:",
        instruction,
        "### Input:",
        input_,
        "### Response:",
        output
    )


def load_data(dataset,
              data_type,
              val_set_size,
              cutoff_len,
              tokenizer):
    if data_type == "alpaca":
        data = TrainSAD(
            dataset,
            val_set_size,
            tokenizer,
            cutoff_len)

    elif data_type == "gpt4all":
        # data = TrainGPT4All(
        #     config.dataset,
        #     config.val_set_size,
        #     tokenizer,
        #     config.cutoff_len)
        raise ValueError(f"Invalid data name: {data_type}")
    else:
        raise ValueError(f"Invalid data name: {data_type}")

    data.prepare_data(use_eos_token=True)
    return data



if __name__ == '__main__':
    is_saving=False
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    ddp = world_size != 1
    device_map = "auto" if not ddp else {"": local_rank}
    print("ddp: ",ddp)
    print("device_map: ",device_map)
    
    if is_saving:

        print("Loading and saving model...")

        # model = AutoModelForCausalLM.from_pretrained(

        #                 'mosaicml/mpt-30b',

        #                 load_in_8bit=False,

        #                 device_map="auto",

        #                 trust_remote_code=True,

        #                 low_cpu_mem_usage=True,

        #                 cache_dir='/cstor/mendeza/hf_test/mpt-30b-cache/'

        #             )

        model = MPTForCausalLM.from_pretrained(

                    'mosaicml/mpt-30b',

                    load_in_8bit=True,

                    device_map='auto',

                    cache_dir=args.model_cache_dir

                )
        model.seqlen = 2048
        model.loaded_in_8bit = True
        # state_dict_contains_metadata = getattr(self, "is_loaded_in_8bit", False)

        model.save_pretrained(args.model_save_dir)
    # Andrew (7.3.2023): There is an issue loading a model from a saved model file
    model = MPTForCausalLM.from_pretrained(

                    'mosaicml/mpt-30b',

                    load_in_8bit=True,

                    device_map=device_map,

                    cache_dir=args.model_cache_dir

                )
    
    print('local_rank: ',local_rank)
    # class MPT7B8bitConfig:

    #     name = 'mpt-7b'

    #     hf_config_name = "mosaicml/mpt-7b"

    #     hf_tokenizer_config = "EleutherAI/gpt-neox-20b"

    #     bits = 8

    #     groupsize = None

    #     max_seq_len = 2048

    #     attn_impl = 'torch'

    #     device_map = "auto"

    # llm_config = MPT7B8bitConfig()

    # model, tokenizer = load_model(llm_config, '/cstor/mendeza/hf_test/mpt-30b-redo', half=False, backend='torch')



    # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b',cache_dir='cstor/mendeza/hf_test/')
    tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b',cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None
    eval_steps=10
    save_steps=50
    save_total_limit=3
    # ddp=False
    lora_out_dir='30b-test-lora'
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["Wqkv"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    for module in model.modules():
            if hasattr(module, 'bias'):
                logger.debug(f'Removing bias ({module.bias}) from {module}.')
                module.bias = None

    model=prepare_model_for_int8_training(model)

    model = get_peft_model(model, lora_config)
    # model.is_parallelizable = True
    # model.model_parallel = True
    model.print_trainable_parameters()
    
    # Apply Gradient checkpointing
    gradient_checkpointing_ratio=1.
    apply_gradient_checkpointing(
                model,
                decoder_layer_class=MPTBlock,
                checkpoint_ratio=gradient_checkpointing_ratio)
    
    training_arguments = transformers.TrainingArguments(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                warmup_steps=5,
                optim="adamw_torch",
                num_train_epochs=1,
                learning_rate=1e-3,
                fp16=True,
                logging_steps=10,
                evaluation_strategy="steps" if eval_steps != 0 else "no",
                save_strategy="steps",
                eval_steps=eval_steps if eval_steps != 0 else None,
                save_steps=save_steps,
                output_dir=lora_out_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if ddp else None,
            )
    data = load_data(dataset=args.dataset_path,
                     data_type='alpaca',
                      val_set_size=50,
                      cutoff_len=256,
                      tokenizer=tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    trainer = transformers.Trainer(
                model=model,
                train_dataset=data.train_data,
                eval_dataset=data.val_data,
                args=training_arguments,
                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
    trainer.train()