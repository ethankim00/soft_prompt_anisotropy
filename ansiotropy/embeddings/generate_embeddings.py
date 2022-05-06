### Extract contextual embeddings from transformer models
from pathlib import Path
from datetime import datetime
from re import template
from typing import Dict
import pickle
from attr import dataclass
import torch
from openprompt.data_utils.utils import InputFeatures
from openprompt import PromptDataLoader
from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification
from openprompt.prompts import ManualVerbalizer


from ansiotropy.data.data_utils import load_validation_data, SUPERGLUE_SCRIPTS_BASE
from openprompt.plms import load_plm
from dataclasses import dataclass


@dataclass
class SoftPromptConfig:
    """Configuration for the soft prompt models"""

    model: str
    model_name_or_path: str
    num_prompt_tokens: int
    initialize_from_vocab: bool


class SoftPromptEmbeddingsExtractor:
    def __init__(self, model_path: str, dataset: str):

        self.model_path = model_path
        self.dataset = dataset

    def __load_from_file(self, file_path: str):
        # Load the model configuration
        # load the parameter state dict from file and initialize the model
        self.plm, self.tokenizer, model_config, WrapperClass = load_plm(
            self.params.model, self.params.model_name_or_path
        )
        self.template = self._load_template(plm, tokenizer)
        self.dataloader = self._load_data(
            self.template, tokenizer=tokenizer, wrapperclass=WrapperClass
        )

        self._load_model()

    def _load_template(self, plm, tokenizer):
        scriptsbase = SUPERGLUE_SCRIPTS_BASE[self.dataset]
        template = SoftTemplate(
            model=plm,
            tokenizer=tokenizer,
            num_tokens=self.params.num_prompt_tokens,
            initialize_from_vocab=self.params.initialize_from_vocab,
        ).from_file(f"scripts/{scriptsbase}/soft_template.txt", choice=0)
        return template

    def _load_verbalizer(self):
        scriptsbase = SUPERGLUE_SCRIPTS_BASE[self.dataset]
        self.verbalizer = ManualVerbalizer(
            self.tokenizer, classes=class_labels
        ).from_file(
            f"scripts/{scriptsbase}/manual_verbalizer.txt",
            choice=0,
        )

    def _load_model(self):
        self.prompt_model = PromptForClassification(
            plm=self.plm,
            template=self.template,
            verbalizer=self.verbalizer,
            freeze_plm=True,
            plm_eval_mode=True,
        )

    def _load_data(self, template, tokenizer, wrapperclass):
        dataset = load_validation_data(self.dataset)
        validation_dataloader = PromptDataLoader(
            dataset=dataset,
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=wrapperclass,
            max_seq_length=500,
            decoder_max_length=3,
            batch_size=16,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail",
        )

        return validation_dataloader

    def extract_soft_prompt_embeddings(prompt_model, inputs):
        prompt_model.eval()
        batch = prompt_model.template.process_batch(inputs)
        batch = {
            key: batch[key]
            for key in batch
            if key in prompt_model.prompt_model.forward_keys
        }
        prompt_model.train()
        outputs = prompt_model.plm(**batch, output_hidden_states=True)
        embeddings = outputs.encoder_hidden_states[0]
        return embeddings.detach().cpu().numpy()

    def save_soft_prompt_embeddings(self, save_dict: bool = True) -> Dict:
        embeddings_dict = {"tokens": {}, "sentences": []}
        self.prompt.prompt_model.eval()
        for step, inputs in enumerate(self.dataloader):
            inputs = inputs.cuda()
            inputs_copy = InputFeatures(**inputs.to_dict()).cuda()
            embeddings = self.extract_soft_prompt_embeddings(
                self.prompt_model, inputs_copy
            )
            for batch_idx in range(inputs["input_ids"].shape[0]):
                for token_idx in range(self.prompt_model.template.num_tokens):
                    embeddings_dict["tokens"]["soft_token_{}".format(token_idx)] = (
                        embeddings_dict["tokens"].get(
                            "soft_token_{}".format(token_idx), []
                        )
                        + embeddings[batch_idx][token_idx].tolist()
                    )
                padding_idx = 0
                for input_id in inputs["input_ids"][batch_idx]:
                    padding_idx += 1
                    if input_id.cpu().numpy() == 0:
                        break
                    token = self.prompt_model.tokenizer.convert_ids_to_tokens(
                        [input_id]
                    )[0].replace("_", "")
                    embeddings_dict["tokens"][token] = embeddings[batch_idx][
                        token_idx + self.prompt_model.template.num_tokens
                    ].tolist()
                sentence = self.prompt_model.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][batch_idx]
                )
                sentence = [
                    token.replace("▁", "") for token in sentence if token != "<pad>"
                ]
                sentence = [
                    "soft_token_{}".format(i)
                    for i in range(self.prompt_model.template.num_tokens)
                ] + sentence
                sentence = " ".join(sentence)
                embeddings_dict["sentences"].append(
                    {sentence: embeddings[batch_idx][:padding_idx]}
                )
        if not experiment_name:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        if save_dict:
            with open(
                Path("./data/embeddings").joinpath(
                    "embeddings_{}.pkl".format(experiment_name)
                ),
                "wb",
            ) as f:
                pickle.dump(embeddings_dict, f)
        embeddings_dict["soft_prompt_tokens"] = self.prompt_model.template.num_tokens
        return embeddings_dict
