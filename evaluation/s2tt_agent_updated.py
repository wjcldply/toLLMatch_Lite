"""
3 ASR classes
    Asr - uses HuggingFace Whisper model (slow)
    AsrJAX - uses Whiseper-JAX either locally or through an API (you need to launch the API `asr_server.py` first)
    AsrOpenaiWhisper - calls OpenAI Whisper API (fast??)
"""

import os, sys, time, argparse, re, logging, copy, pdb, msgpack
from termcolor import cprint
from dotenv import load_dotenv
from typing import Optional, List
from argparse import Namespace
from collections import Counter
import numpy as np
import torch, requests, json
from termcolor import cprint
from transformers import AutoTokenizer


load_dotenv("../.env", override=True)  # load API keys into
print(os.getenv("HF_HOME"))
sys.path.append("../")

import jax.numpy as jnp
from jax import jit
from whisper_jax import FlaxWhisperForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from utils.utils import update_source_word_list, purge_directory, parse_language_pair
from chat_templates.templates import PROMPT_TEMPLATES


from vllm import LLM, SamplingParams

import io
from openai import OpenAI
from utils.utils import Timer

from scipy.io.wavfile import write as wav_write

# from configs import config_1 as CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--config_id", type=int, default=-1)
parser.add_argument("--model_id", type=str, default="")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--use_api", action="store_true")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--k", type=int, default=4)
parser.add_argument("--dir", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--use_asr_api", action="store_true")
parser.add_argument("--asr_model_size", type=str, default="small")
parser.add_argument("--prompt_id", type=int, default=0)
parser.add_argument("--bgd_info", action="store_true")
parser.add_argument("--min_read_time", type=float, default=0)
parser.add_argument("--min_lag_words", type=int, default=1)
parser.add_argument("--func_wrds", type=str, default="[]")
parser.add_argument("--priming", action="store_true")
args, unknown_args = parser.parse_known_args()

from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction


# --------------------------------------------------------------------------------------------------------------------
purge_directory(args.output)
DEVICE = args.device
WAIT_K = args.k
verbose = args.verbose
use_asr_api = args.use_asr_api
PROMPT_ID = args.prompt_id
ENDPOINT = os.environ["VLLM_SERVER_ENDPOINT_URL"]
ASR_ENDPOINT = os.environ["ASR_SERVER_ENDPOINT_URL"]
print(ENDPOINT)
print(ASR_ENDPOINT)

ASR_MODEL_SIZE = args.asr_model_size

# FIXME: these should be CLI arguments
ASR_MODEL_NAME = f"openai/whisper-{ASR_MODEL_SIZE}.en"  # "openai/whisper-tiny.en" # "openai/whisper-small.en" "openai/whisper-large-v2"
if ASR_MODEL_SIZE == "large-v3":
    ASR_MODEL_NAME = "openai/whisper-large-v3"
if ASR_MODEL_SIZE == "distil-large-v3":
    ASR_MODEL_NAME = "distil-whisper/distil-large-v3"

SRATE = 16000
MIN_LAG_WORDS = int(args.min_lag_words)
MIN_READ_TIME = float(args.min_read_time)
RESPONSE_PRIMING = bool(args.priming)
cprint(f"RESPONSE_PRIMING: {RESPONSE_PRIMING}", "black", "on_light_magenta")

if args.model_id == "meta-llama/Llama-2-13b-chat-hf":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
elif args.model_id == "microsoft/Orca-2-7b":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
elif args.model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "meta-llama/Meta-Llama-3-70B-Instruct":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "casperhansen/llama-3-8b-instruct-awq":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "casperhansen/llama-3-70b-instruct-awq":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "Ġ"
    EOT_TOKEN_SEQUENCE = ["<|eot_id|>"]
elif args.model_id == "microsoft/Phi-3-mini-4k-instruct":
    ACCEPTS_SYSTEM_MESSAGE = True
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["<|end|>", "<|endoftext|>"]
elif args.model_id == "google/gemma-7b-it":
    ACCEPTS_SYSTEM_MESSAGE = False
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["<end_of_turn>", "<eos>"]
elif args.model_id == "mistralai/Mistral-7B-Instruct-v0.2":
    ACCEPTS_SYSTEM_MESSAGE = False
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
elif args.model_id == "mistralai/Mistral-7B-Instruct-v0.1":
    ACCEPTS_SYSTEM_MESSAGE = False
    TOKENIZATION_SPACE = "▁"
    EOT_TOKEN_SEQUENCE = ["[/INST]", "</s>"]
else:
    raise RuntimeError("Unknown model id")
model_id = args.model_id

# function_words = [
#     "the",
#     "a",
#     "is",
#     "am",
#     "in",
#     "out",
#     "by",
#     "on",
#     "off",
#     "down",
#     "up",
#     "off",
#     "and",
#     "will",
#     "to",
#     "from",
#     "not",
# ]
# function_words = ["the", "a", "is", "am", "to", "will", "not"]
# function_words = []

cprint(args.func_wrds, "red", "on_cyan")
function_words = args.func_wrds.split("_") if not args.func_wrds == "_" else []
cprint(f"function_words: {function_words}", "red", "on_cyan")
time.sleep(2)


# --------------------------------------------------------------------------------------------------------------------

SRC_LANG, TARG_LANG = parse_language_pair(args.dir)

A = []  # each time step마다 소요된 generation time 저장됨


def check_if_asr_model_is_right(asr_model_size: str) -> None:
    response = requests.post(f"{ASR_ENDPOINT}/info", json={})
    model_running_on_api = json.loads(response.text)["asr_model_name"]
    cprint(
        f"ASR model running at API: {model_running_on_api}",
        "black",
        "on_red",
        attrs=["bold"],
    )
    assert model_running_on_api == ASR_MODEL_NAME, "Wrong ASR model running at API."


# llm = LLM(model=model_id, max_num_seqs=1, max_model_len=4096, block_size=8)
if not args.use_api:
    tensor_parallel_size = 1  # torch.torch.cuda.device_count()
    llm = LLM(model=model_id, max_num_seqs=1, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
else:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    check_if_asr_model_is_right(ASR_MODEL_SIZE)


def get_last_word(outputs, prompt):
    '''
    outputs: list of dictionaries(each with 'generated_text' key and corresponding String value)
    '''
    return list(
        map(
            # 모델이 생성한 답안 중 마지막 단어 제외하는 로직
            lambda x: " ".join(x["generated_text"][len(prompt) :].split(" ")[:-1]),  # each dict in list의 'generated_text' key에 대응되는 String value를 " "로 split하여 단어별로 리스트에 담았다가 마지막 요소를 제외하여 다시 join한 String으로 저장
            outputs,
        )
    )


def build_full_prompt(partial_source, partial_target, background=None):
    ############################################################################
    if isinstance(background, dict):  # If background is a dictionary, convert it to a formatted string
        background_str = ", ".join(f"{k}: {v}" for k, v in background.items())
    else:
        background_str = background or ""
    ############################################################################

    DEFAULT_SYSTEM_PROMPT = PROMPT_TEMPLATES[PROMPT_ID](
        SRC_LANG=SRC_LANG, 
        TARG_LANG=TARG_LANG, 
        ############################################################################
        # BGD_INFO=background
        BGD_INFO=background_str  # Use the formatted background string
        ############################################################################
    )

    if ACCEPTS_SYSTEM_MESSAGE:  # 시스템메세지 따로 받는 모델인 경우에 대응하기 위함
        messages = [
            {"role": "system", "content": f"{DEFAULT_SYSTEM_PROMPT}"},  # 시스템 프롬프트는 'system' role로
            {"role": "user", "content": f"context: {partial_source}"},  # 번역 전 부분적 입력은 'user' role로
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": f"{DEFAULT_SYSTEM_PROMPT}\nContext: {partial_source}",
            }
        ]

    # system prompt + partial input을 토크나이즈해 tmp_prompt로
    tmp_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if RESPONSE_PRIMING:  # Response Priming(partial target을 assistnat role로 프롬프팅 하는 방식 활용하는 경우)
        prompt = f"{tmp_prompt}{TARG_LANG} translation:{partial_target}"  # 최종 prompt는 (system prompt + partial input + "TGT_LANG translation:" +partial target)로 이뤄짐
    else:
        prompt = f"{tmp_prompt}{partial_target}"
    return re.sub(r"\s+([,.!?;:])", r"\1", prompt)


sampling_params = SamplingParams(
    temperature=0,
    min_tokens=2,
    max_tokens=20,
    stop=[" "],
    include_stop_str_in_output=True,
    # use_beam_search=True,
    # best_of=3,
    # top_p=0.95,
)

normal_generation_params = SamplingParams(
    temperature=0.5,
    min_tokens=2,
    max_tokens=100,
    stop=[";"],
    include_stop_str_in_output=True,
    # use_beam_search=True,
    # best_of=3,
    top_p=0.95,
)


class Translator:

    def __init__(self, generation_kwargs, function_words):
        self.partial_target = []  # 실시간으로 번역된 타겟언어 번역내용 리스트 (스플릿해서 담아두는듯?)
        self.flags = dict(consecutive_duplicates=0)  # no idea
        self.generation_kwargs = generation_kwargs  # generation arguments to pass to LLMs
        self.FUNCTION_WORDS = function_words  # words w/ no semantic/lexical purpose (e.g. a, the, umm, etc.)
        self.background = None
        
        ############################################################################
        self.history_words = None
        self.background_dict = {
            "topic": "",
            "named_entities": []
        }
        self.background_subjects_set = set()
        self.MAX_NAMED_ENTITIES = 50  # Limit total named entities
        self.MAX_NEW_ENTITIES_PER_UPDATE = 2  # Maximum new entities to add per update
        ############################################################################

    ############################################################################
    def generate_background_info(self, new_source_context):
        """
        Generate comprehensive background information for the entire context in two steps:
        1. Generate a topic string.
        2. Generate named entities.
        """
        
        # Step 1: Generate a simple topic string
        topic_prompt = f"""
        You are an advanced knowledge extraction AI specializing in comprehensive contextual analysis.
        Generate a concise(under 15 words) overarching topic from the given transcript.
        
        Transcript
        "{' '.join(self.history_words)} {new_source_context}"

        IMPORTANT
        Answer with just the topic, and Nothing more. Don't Explain or Elaborate. Simply write the topic as it is.
        When answering, Don't add notes or explanations and stick to formats.
        
        Topic you suggest should follow formats below.
        Answer: Specific Explanation of Core Idea of Content from Transcript;

        If There are not enough information to conclude the topic, simply answer with a single semicolon like below.
        Answer: ;
        
        """

        try:
            # Generate response for the topic
            # print(f"TR PROMPT: {topic_prompt}")
            topic_response = self.___generate(topic_prompt).lstrip()
            if topic_response.startswith("Answer: "):
                topic_response = topic_response[len("Answer: "):-1]
            topic = topic_response.strip()
            print(f"TR: {topic}")

            # Step 2: Generate named entities
            named_entities_prompt = f"""
            You are an advanced knowledge extraction AI specializing in entity recognition and description.
            Identify up to 3 specific,challenging,significant entities that are either mentioned OR highly related to the given transcript.
            Find at least 1 entity that's not mentioned in Transcript explicitly, but is likely to be brought up soon.
            Include figures, places, concepts, key personalities, organizations, etc. 
            For each entity, provide a concise, informative one-line description.
            Don't answer with Entities that are already identified.

            Already Identified Entities are {'/'.join([entity['entity'] for entity in self.background_dict.get('named_entities', [])])}
            
            Transcript
            "{' '.join(self.history_words)} {new_source_context}"
            
            Answer in the format of each entity and description listed in alternation, with / separating each and a semicolon at the end.
            When answering, Don't add notes or explanations. Stick to formats.
            
            Your Answers MUST follow formats below.
            Answer: entity1/description of entity1/entity2/description of entity2;

            If There are nothing to identify, simply answer with a single semicolon like below.
            Answer: ;
            """
            """Existing Named Entities: {', '.join([entity['entity'] for entity in self.background_dict.get('named_entities', [])])}"""

            # Generate response for named entities
            # print(f"NER PROMPT: {named_entities_prompt}")
            named_entities_response = self.___generate(named_entities_prompt).strip()
            if named_entities_response != "":
                if named_entities_response.startswith("Answer: "):
                    named_entities_response = named_entities_response[len("Answer: "):-1]
                if named_entities_response.endswith("/"):
                    named_entities_response = named_entities_response[:-1]
                named_entities = named_entities_response.lower().strip().split("/")
                print(f"NER: {named_entities_response}")

                # Limit the number of named entities
                named_entities = named_entities[:self.MAX_NEW_ENTITIES_PER_UPDATE*2]
                # Construct the final background dictionary
                final_background = {
                    "topic": topic,
                    # "named_entities": [{"entity": entity, "description": "Description not available"} for entity in named_entities]
                    # "named_entities": named_entities
                    "named_entities" : [
                        {"entity": named_entities[i], "description": named_entities[i + 1]}
                        for i in range(0, len(named_entities), 2)
                    ]
                }
            else:
                named_entities = None
                # Construct the final background dictionary
                final_background = {
                    "topic": topic,
                    "named_entities" : []
                }
            return final_background
        except Exception as e:
            print(f"Error generating background info: {e}")
            return {"topic": "", "named_entities": []}

    def update_background_info(self, new_source_words):
        """
        Update background information dictionary
        """
        # Filter out function words
        filtered_words = [
            word for word in new_source_words 
            if word.lower() not in self.FUNCTION_WORDS
        ]

        # If no new meaningful words, skip update
        if not filtered_words:
            return

        # Join filtered words into a single context string
        new_source_context = " ".join(filtered_words)

        # Generate new background information
        new_background = self.generate_background_info(new_source_context)

        # Update topic if not empty
        if new_background.get("topic"):
            self.background_dict["topic"] = new_background["topic"]

        # Update named entities
        for new_entity in new_background.get("named_entities", []):
            new_entity['entity'] = new_entity['entity'].strip()
            # Check if entity already exists
            if not any(
                existing['entity'] == new_entity['entity'] or new_entity['entity'] in existing['entity'] or existing['entity'] in new_entity['entity']
                for existing in self.background_dict['named_entities']
            ):
                if ":" not in new_entity['entity'] and len(new_entity['entity'])<100:
                    # Ensure we don't exceed max entities
                    if len(self.background_dict['named_entities']) >= self.MAX_NAMED_ENTITIES:
                        # Remove the oldest entity
                        self.background_dict['named_entities'].pop(0)
                    
                    # Add new entity
                    self.background_dict['named_entities'].append(new_entity)

        # Update the background as a string for compatibility
        self.background = str(self.background_dict)
    # def generate_background_info(self, new_source_context):
    #     """
    #     Generate comprehensive background information for the entire context
    #     """
    #     # Prepare existing context from current background
    #     existing_entities = "; ".join(
    #         f"{entity['entity']}: {entity['description']}" 
    #         for entity in self.background_dict.get("named_entities", [])
    #     )
        
    #     # Comprehensive prompt for extracting topic and named entities
    #     background_prompt = f"""
    #     You are an advanced knowledge extraction AI specializing in comprehensive contextual analysis.

    #     Existing Context:
    #     Topic: {self.background_dict.get('topic', 'N/A')}
    #     Existing Entities: {existing_entities}

    #     New Input Context:
    #     {new_source_context}

    #     Your Complex Tasks:
    #     1. Generate an Overarching Topic:
    #        - Craft a concise, intellectually engaging topic that encapsulates 
    #          the broader thematic connections in the given context
    #        - Must be broader than the specific input
    #        - Maximum 12-15 words

    #     2. Identify NEW Named Entities:
    #        - Select up to 3 most significant entities NOT already in existing context
    #        - Include historical figures, places, concepts, or key personalities
    #        - Ensure entities are contextually relevant and intellectually meaningful

    #     3. For Each Entity:
    #        - Provide a crisp, informative one-line description
    #        - Focus on their core significance, historical impact, or conceptual importance
    #        - Maximum 15 words per description
    #     """
    #     background_prompt += """
    #     Output Format (Strict JSON):
    #     {
    #         "topic": "Overarching Intellectual Theme",
    #         "named_entities": [
    #             {
    #                 "entity": "Entity Name",
    #                 "description": "Concise, Meaningful Description"
    #             },
    #             ...
    #         ]
    #     }
    #     """

    #     # Perform single LLM inference
    #     try:
    #         import json
            
    #         # Generate response
    #         response = self.__generate(background_prompt)
            
    #         # Attempt to parse JSON
    #         parsed_response = json.loads(response)
            
    #         return parsed_response
    #     except Exception as e:
    #         print(f"Error generating background info: {e}")
    #         return {"topic": "", "named_entities": []}

    # def update_background_info(self, new_source_words):
    #     """
    #     Update background information dictionary
    #     """
    #     # Filter out function words
    #     filtered_words = [
    #         word for word in new_source_words 
    #         if word.lower() not in self.FUNCTION_WORDS
    #     ]

    #     # If no new meaningful words, skip update
    #     if not filtered_words:
    #         return

    #     # Join filtered words into a single context string
    #     new_source_context = " ".join(filtered_words)

    #     # Generate new background information
    #     new_background = self.generate_background_info(new_source_context)

    #     # Update topic if not empty
    #     if new_background.get("topic"):
    #         self.background_dict["topic"] = new_background["topic"]

    #     # Update named entities
    #     for new_entity in new_background.get("named_entities", []):
    #         # Check if entity already exists
    #         if not any(
    #             existing['entity'] == new_entity['entity'] 
    #             for existing in self.background_dict['named_entities']
    #         ):
    #             # Ensure we don't exceed max entities
    #             if len(self.background_dict['named_entities']) >= self.MAX_NAMED_ENTITIES:
    #                 # Remove the oldest entity
    #                 self.background_dict['named_entities'].pop(0)
                
    #             # Add new entity
    #             self.background_dict['named_entities'].append(new_entity)

    #     # Update the background as a string for compatibility
    #     self.background = str(self.background_dict)
    def ___generate(self, prompt, get_more_tokens=False):
        if args.use_api:
            prompt_len = len(prompt)
            payload = {"prompt": prompt}
            payload.update(
                {
                    k: v
                    for k, v in normal_generation_params.__dict__.items()
                    if k
                    in [
                        "temperature",
                        "min_tokens",
                        "max_tokens",
                        "stop",
                        "include_stop_str_in_output",
                        # "use_beam_search",
                        # "best_of",
                    ]
                }
            )

            # set min_tokens to 3 when requested (e.g. a single whitespace is generated)
            payload["min_tokens"] = 3 if get_more_tokens else normal_generation_params.min_tokens
            response = requests.post(ENDPOINT, json=payload)
            # pdb.set_trace()
            return json.loads(response.text)["text"][0][prompt_len:].replace("\n", "")  # generation 결과물에서 new-line 없애줌

        else:
            result = llm.generate(prompts=[prompt], use_tqdm=False, sampling_params=normal_generation_params)  # 걍 프롬프트로 generate()
            return result[0].outputs[0].text.replace("\n", "")  # generation 결과물에서 new-line 없애줌
    ############################################################################


    def __generate(self, prompt, get_more_tokens=False):
        if args.use_api:
            prompt_len = len(prompt)
            payload = {"prompt": prompt}
            payload.update(
                {
                    k: v
                    for k, v in self.generation_kwargs.__dict__.items()
                    if k
                    in [
                        "temperature",
                        "min_tokens",
                        "max_tokens",
                        "stop",
                        "include_stop_str_in_output",
                        # "use_beam_search",
                        # "best_of",
                    ]
                }
            )

            # set min_tokens to 3 when requested (e.g. a single whitespace is generated)
            payload["min_tokens"] = 3 if get_more_tokens else self.generation_kwargs.min_tokens
            response = requests.post(ENDPOINT, json=payload)
            # pdb.set_trace()
            return json.loads(response.text)["text"][0][prompt_len:].replace("\n", "")  # generation 결과물에서 new-line 없애줌

        else:
            result = llm.generate(prompts=[prompt], use_tqdm=False, sampling_params=self.generation_kwargs)  # 걍 프롬프트로 generate()
            return result[0].outputs[0].text.replace("\n", "")  # generation 결과물에서 new-line 없애줌

    def _generate(self, prompt):

        just_generated = self.__generate(prompt)  # 위 함수로 프롬프트에 대한 생성 수행

        # NOTE: EXPERIMENTAL. Handles the edge case where the LLM doesn't prepend a space its output
        if not just_generated.endswith(" "):
            just_generated = f"{just_generated} "

        if just_generated == " ":  # NOTE: if only a space is generated, try again with more tokens
            just_generated = self.__generate(prompt, get_more_tokens=True)

        for eot_token_string in EOT_TOKEN_SEQUENCE:  # eot, eos, </s> 등 생성완료 토큰이 있는지 체크
            if eot_token_string in just_generated:  # 만약 생성완료 토큰이 나온 경우라면
                self.flags["stop_reason"] = "eot_id"  # 클래스 인스턴스 플래스 값으로 생성완료 이유(토큰 그대로 쓰네) 넣고
                last_word = just_generated.split(eot_token_string)[0]  # 마지막 생성된 단어 가져오고 (code 자체는 string 가져오게 되어있긴 한데 동작할 땐 partial translated까지만 입력으로 들어가고 출력은 partial source - partial translated에 해당하는 내용인 (기번역되지 않아있던) 마지막 단어만 나오므로 last sentence가 아닌 last word 변수명을 지정한 것으로 보임)
                self.flags["last_word"] = last_word  # 마지막 생성단어 플래그에 넣고
                return True  # True 반환하면서 _generate() 종료

        # detect if <|start_header_id|> is generated, chop off what's after it (Llama-3 + ASR specific)
        if "<|start_header_id|>" in just_generated:  # 라마3 계열에서 <|start_header_id|> 토큰이 생성되는 경우에 대응
            self.flags["stop_reason"] = just_generated.split("<|start_header_id|>")[0]  # WTF
            # self.flags["last_word"] = just_generated.split("<|start_header_id|>")[0]  # 뭔가 잘못된거같아서 수정함
            self.flags["stop_reason"] = "new_word"
            return True

        # detect if full stop, question or exclamation mark is generated, chop off what's after it
        just_generated, full_stop_generated = self._check_if_full_stop(just_generated)
        if full_stop_generated:  # 번역할 내용인 source가 모두 완료되어 정지된 경우
            self.flags["stop_reason"] = "new_word"
            self.flags["last_word"] = just_generated
            return True

        if " " in just_generated[1:]:
            self.flags["stop_reason"] = "new_word"
            self.flags["last_word"] = just_generated.split(" ")[-2]  # 펑추에이션 마크와 _check_if_full_stop()에서 임의로 추가된 공백문자 제외한 문자열(단어)를 플래그 last_word로 저장
            return True

        if self.flags["stop_reason"] is None:
            return True

        else:
            raise RuntimeError("Wrong stop reason.")

    def _step(self, partial_source, partial_target):
        '''
        다음 Translation Step 넘어가도록 프롬프트 업데이트 + 번역생성 수행
        '''
        self.flags["partial_target"] = partial_target

        prompt = build_full_prompt(partial_source, partial_target, background=self.background)

        if verbose:
            cprint(prompt, color="green")

        print("in the context: ", self.generation_kwargs.temperature)  # WTF
        st = time.time()
        _ = self._generate(prompt)   # 업데이트된 프롬프트로 생성 수행
        A.append(time.time() - st)  # A란 리스트에 생성스텝에 소요된 시간 append

        print("outside the context: ", self.generation_kwargs.temperature)  # WTF

        if verbose:
            cprint(f"stop_reason: {self.flags['stop_reason']}", color="magenta")

    def _check_if_full_stop(self, string):
        '''
        입력 string에 . ? ! : 와 같은 마크가 들어있는지 확인함
        만약 마크 포함되어 있는데 인스턴스 플래그 source_finished를 통해 번역할 내용이 더 이상 없음이 확인될 경우 True값을 반환
        '''
        punctuation_marks = {".", "?", "!", ":"}
        for char in string:
            if char in punctuation_marks:
                string = string.split(char)[0]  # take the string before the FIRST punctuation mark
                if self.flags["source_finished"]:
                    return f"{string}{char}", True  # true EOS detected
                else:  # just insert a space before the punctuation mark, it will be dropped in _generate
                    return f"{string} {char}", False  # 아직 원문 번역 안 끝난 경우에 해당하며, 펑추에이션 마크 앞에 공백 추가하여 _generate() 함수에서 처리함
        return (
            string,
            False,
        )  # if no punctuation mark is found, return the original string

    def _translate(self, partial_source_lst: List, source_finished: bool, k=4):
        '''
        실제 번역 수행하는 핵심 메소드
        '''

        ############################################################################
        self.update_background_info(partial_source_lst)
        ############################################################################

        self.flags["source_finished"] = source_finished  # 원문 번역 완료 여부
        self.flags["stop_reason"] = None  # _generat() 정지이유 플래그
        self.flags["last_word"] = ""  # 기생성된 마지막 단어

        # if the new source word is a function word, READ action
        if (not source_finished) and (partial_source_lst[-1].lower() in self.FUNCTION_WORDS):  # 원문번역완료X || 새로 ASR받은 Partial Source 최신 단어가 Function Words에 속하면 그냥 다음 Partial Source 받기
            return dict(action="READ", content="", final=False)  # 딕셔너리 반환이네???

        # otherwise do the full cycle
        partial_source = " ".join(partial_source_lst)  # 이미 번역된 기존 Partial Source 리스트 내의 각 단어들 공백 넣어 Join한 String 만들기
        partial_source = re.sub(r"\s+([,.!?;:])", r"\1", partial_source)  # (Partial Source) 펑추에이션 마크 앞에 공백이 있으면 삭제
        self.partial_target = [re.sub(r"\s+([,.!?;:])\s+", r"\1", it) for it in self.partial_target]  # (Partial Target) 펑추에이션 마크 앞에 공백이 있으면 삭제

        partial_target = re.sub(r"\s+", " ", " ".join(self.partial_target))  # (Partial Target) 다중 스페이스 발견하면 단일 스페이스로 치환

        if verbose:
            with open(f"{args.output}/inc_asr.log", "a") as f:
                f.write(partial_source + "\n")
            with open(f"{args.output}/inc_trasnation.log", "a") as f:
                f.write(partial_target + "\n")

        # (업데이트된) Partial Source / (새로운 단어 번역 안 된) Partial Target으로 다음 스텝 번역 Generation 수행
        self._step(partial_source, partial_target)

        if verbose:
            print(f"{partial_source}|{partial_target}|{self.flags['last_word']}|")
            print("-" * 89)

        # read more if the last genrated word is a duplicate of the previous generated word, otherwise append to target
        if len(self.partial_target) > 0:  # 새로 번역한 단어가 있는 경우
            if self.partial_target[-1].lower() == self.flags["last_word"].lower():  # 앞 스텝에서 이미 만들었던 단어를 다시 반복한 것이라면
                self.flags["consecutive_duplicates"] += 1  # 인스턴스 플래그의 consecutive duplicates 증가시킴
                if verbose:
                    print(f"{self.flags['consecutive_duplicates']} consecutive duplicates detected")
            else:  # 앞 스텝에서 이미 만들었던 단어가 아닌 새로운 단어를 번역해낸 것이라면
                self.flags["consecutive_duplicates"] = 0  # 인스턴스 플래그의 consecutive duplicates 0으로 초기화

        if self.flags["consecutive_duplicates"] > 2:  # 2회 연속으로 동일 단어 번역생성한 경우엔
            return dict(action="READ", content="", final=False)  # 추가 ASR 입력 받아오기

        # decide whether to append a new word (update the paratial_target)
        if self.flags["stop_reason"] in ["new_word", "eot_id"]:
            self.flags["last_word"], full_stop_detected = self._check_if_full_stop(self.flags["last_word"])
            if full_stop_detected and source_finished:   # 원문의 번역이 완전히 끝난 경우
                self.partial_target.append(self.flags["last_word"])  # partial target에 새로운 단어 추가해 업데이트
                return dict(action="WRITE", content=self.flags["last_word"], final=True)  # 마지막 번역단어 WRITE Action 수행하고, Final:True 반환
        if self.flags["stop_reason"] == "new_word":  # 완전히 번역이 끝나진 않았고 새로운 단어 번역이 완성되어 중지된 경우엔
            self.partial_target.append(self.flags["last_word"])  # partial target에 새로운 단어 추가해 업데이트
            return dict(action="WRITE", content=self.flags["last_word"], final=False)  # 마지막 번역단어 WRITE Action 수행하고, Final:False 반환
        elif self.flags["stop_reason"] == "eot_id":  # 새로운 단어 번역이 완성되어서 중단된 것이 아닌, eot 토큰이 나와서 중단된 경우엔
            if source_finished:  # 번역할 소스 원문이 완전히 번역된 경우엔
                self.partial_target.append(self.flags["last_word"])  # partial target에 새로운 단어 추가해 업데이트
                return dict(action="WRITE", content=self.flags["last_word"], final=True)  # 마지막 번역단어 WRITE Action 수행하고, Final:True 반환
            else:  # 아직 번역할 소스 원문이 남은 경우엔
                return dict(action="WRITE", content="", final=False)  # Null String으로 WRITE Action 수행하고, Final:False 반환
        else:
            # raise RuntimeError('asdfasd')
            cprint("******** INVALID STOP REASON *********", "black", "on_yellow")
            return dict(action="READ", content="", final=False)


class AsrJAX:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)

        self.srate = SRATE
        self.DEVICE = DEVICE

        if not use_asr_api:
            self.model = FlaxWhisperForConditionalGeneration.from_pretrained(
                ASR_MODEL_NAME, dtype=jnp.bfloat16, _do_init=True
            )

            self.p_generate = jit(self.generate_fn)
            self._warmup_asr()
        else:
            pass

    def _warmup_asr(self):
        cprint("Warming up the ASR model...", "black", "on_cyan", attrs=["bold"])
        input_features = self.processor(
            np.random.randn(32000), sampling_rate=self.srate, return_tensors="np"
        ).input_features
        self.p_generate(input_features)

    def generate_fn(self, input_features):
        pred_ids = self.model.generate(
            input_features,
            task="transcribe",
            return_timestamps=False,
            max_length=self.model.config.max_length,
            params=self.model.params,
        )
        return pred_ids.sequences

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:
        if not use_asr_api:
            input_features = self.processor(audio_array, sampling_rate=self.srate, return_tensors="np").input_features
            predicted_ids = self.p_generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        else:
            # NOTE: slow because we're sending float64 over JSON
            # response = requests.post(
            #     f"{ASR_ENDPOINT}/generate",
            #     json={"source": audio_array, "source_finished": source_finished}
            # )
            # NOTE: faster because we're sending float16 over msgpack
            response = requests.post(
                f"{ASR_ENDPOINT}/generate",
                data=msgpack.packb(
                    {
                        "source": np.array(audio_array).astype(np.float16).tolist(),
                        "source_finished": source_finished,
                    }
                ),
                headers={"Content-Type": "application/msgpack"},
            )
            transcription = json.loads(response.text)["recognized_word_list"]
        return self._postprocess(transcription, source_finished)


class Asr:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME, dtype=torch.float16)
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL_NAME,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.asr_model.config.forced_decoder_ids = None
        self.srate = SRATE
        self.DEVICE = DEVICE

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:
        input_features = self.processor(
            audio_array,
            sampling_rate=self.srate,
            return_tensors="pt",
            device=f"cuda:{self.DEVICE}",
            dtype=torch.float16,
            pad_to_multiple_of=128,
        ).input_features.to(device=f"cuda:{self.DEVICE}", dtype=torch.half)
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        return self._postprocess(transcription, source_finished)


class AsrOpenaiWhisper:

    def __init__(self, ASR_MODEL_NAME, DEVICE, SRATE):
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)

        self.srate = SRATE
        self.DEVICE = DEVICE

        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY_MINE"))

    def _postprocess(self, s, source_finished):
        # drop incomplete words
        s = [i for i in s.split(" ") if not (i.endswith("-") or i.endswith("..."))]
        if len(s) == 0:
            return []
        if source_finished:  # NOTE: we only return all the sourc words when the source is finished
            if not s[-1].endswith("."):
                s[-1] += "."
            return s
        else:
            s = [i.replace(".", ",") for i in s]
            if len(s) > 0:
                s[0] = s[0][0].upper() + s[0][1:]
            if len(s) > 1:
                for i in range(len(s) - 1):
                    if s[i].endswith(","):
                        if not s[i + 1].startswith("I"):
                            s[i + 1] = s[i + 1].lower()
            return s[:-MIN_LAG_WORDS]

    def recognize(self, audio_array: list, source_finished: bool) -> List[str]:

        cprint("Using OpenAI Whisper ASR", "black", "on_yellow", attrs=["bold"])
        buffer = io.BytesIO()
        buffer.name = "audio.wav"
        wav_write(buffer, SRATE, np.array(audio_array))
        buffer.seek(0)
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,  # Use the in-memory file-like object
            response_format="text",
            # prompt="ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T."
        )
        return self._postprocess(transcription, source_finished)


@entrypoint
class Agent(SpeechToTextAgent):

    def __init__(self, args: Optional[Namespace] = None) -> None:
        super().__init__(args)
        self.generation_kwargs = sampling_params
        self.function_words = function_words
        # self.asr_model = Asr(ASR_MODEL_NAME, DEVICE, SRATE)
        self.asr_model = AsrJAX(ASR_MODEL_NAME, DEVICE, SRATE)
        # self.asr_model = AsrOpenaiWhisper(ASR_MODEL_NAME, DEVICE, SRATE)
        self.deduplicated_list_of_words = []
        self.history_list_of_words = []
        self.background_dict_backup = {
            "topic": "",
            "named_entities": []
        }
        self._reset()

    def _set_background(self, background):
        """the evaluator sets it when path to backgrounds is provided as a CLI argument"""
        self.translator.background = background  # Background Information을 Translator LLM 인스턴스에 저장
        self.translator.history_words = self.history_list_of_words  # 지금까지 ASR로 전달받은 전체 입력 단어들 리스트

    def _save_asr_and_translation(self):
        with open(f"{args.output}/asr.log", "a") as f:
            f.write(" ".join(self.deduplicated_list_of_words) + "\n")
        with open(f"{args.output}/translation.log", "a") as f:
            f.write(" ".join(self.translator.partial_target) + "\n")
        self.background_dict_backup = self.translator.background_dict

    def _reset(self):
        if verbose:
            cprint("resetting translator", color="red", attrs=["bold"])
        self.history_list_of_words.extend(self.deduplicated_list_of_words)
        # Translator LLM 인스턴스 변수 선언
        self.translator = Translator(function_words=self.function_words, generation_kwargs=self.generation_kwargs)
        self.translator.background_dict = self.background_dict_backup

        self.deduplicated_list_of_words = []  # 새로 ASR 해온 단어들이 차곡차곡 들어가는 리스트임
        self.first_batch = True  # WTF

    def policy(self):

        # NOTE: EXPERIMENTAL. Unconditionally return a READ action if the source is < 2.4 s
        if not self.states.source_finished:
            if len(self.states.source) / SRATE <= MIN_READ_TIME:
                return ReadAction()

        # FIXME: add docstring what does it do?
        recognized_word_list = self.asr_model.recognize(
            self.states.source,  # NOTE: we only take the last 3 seconds of audio
            self.states.source_finished,
        )

        # if no words are recognized yet (e.g. at the start of a sentence), return a READ action
        if len(recognized_word_list) == 0:
            return ReadAction()

        # add fresh input words to the existing list of input words, without duplicates
        _updated_source_word_list, num_words_added = update_source_word_list(
            self.deduplicated_list_of_words, recognized_word_list, verbose=verbose
        )

        # if no new words are added or even fewer than before, return a READ action
        if not self.states.source_finished:
            if len(_updated_source_word_list) <= len(self.deduplicated_list_of_words):
                cprint("No new words added", "white", "on_red", attrs=["bold"])
                return ReadAction()

        self.deduplicated_list_of_words = _updated_source_word_list

        # FIXME: experimental (ASR is only allowed to add one or zero words at a time)
        if not self.states.source_finished and not self.first_batch:
            lim = len(self.deduplicated_list_of_words) - max(1, num_words_added) + 1
            self.deduplicated_list_of_words = self.deduplicated_list_of_words[:lim]  # ASR로 새로 추가해주는 소스 단어가 1개를 넘지 않도록 보정해주는 과정

        if (len(self.deduplicated_list_of_words) <= WAIT_K) and (not self.states.source_finished):
            return ReadAction()

        # Translator LLM 인스턴스의 _translate() 호출 : 새로 ASR한 단어가 추가된 List, 지금까지 번역된 내용이 담긴 List, 대기 k를 전달
        result = self.translator._translate(self.deduplicated_list_of_words, self.states.source_finished, k=WAIT_K)  # _translate()은 번역 결과를 Dict로 반환함
        self.first_batch = False  # clear the first batch (of audio chunks) flag

        # cprint(self.deduplicated_list_of_words, color='cyan', attrs=['bold'])
        # if self.states.source_finished:
        #     return WriteAction('asdf', finished=True)
        # else:
        #     return WriteAction('asdf', finished=False)

        # if the source is finished, keep translating until the model outputs a final translation
        if self.states.source_finished:
            _result = ""
            while not result["final"]:  # _generate() 결과 Dict의 Final:False인 경우 (아직 번역완료 아닌경우)
                # 계속 루프 돌면서 번역 완료할때까지 대기
                result = self.translator._translate(
                    self.deduplicated_list_of_words,
                    self.states.source_finished,
                    k=WAIT_K,
                )
                _result = " ".join([_result, result["content"]])
            result["content"] = _result

        if verbose:
            cprint(f"{np.mean(A):.2f}, {np.std(A):.2f} N: {len(A)}", color="green")
            cprint(result, color="yellow")
            cprint(len(self.states.source), color="yellow")
            cprint(self.deduplicated_list_of_words, color="cyan", attrs=["bold"])

        if result["action"] == "READ":
            return ReadAction()
        elif result["action"] == "WRITE":
            if result["final"]:
                self._save_asr_and_translation()
                self._reset()
            return WriteAction(result["content"], finished=result["final"])
        else:
            raise RuntimeError("Policy Error: Unknown Action")
