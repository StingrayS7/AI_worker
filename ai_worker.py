# -*- coding: utf-8 -*-
"""

Original file is located at
    https://colab.research.google.com/drive/1kRMztLQaPMI2uAIOsakUacCOMyM1iy1L

---

# 1.   Устанавливаем библиотеки для трассировки с помощью pheonix
"""

# Устанавливаем библиотеки для трассировки с помощью pheonix
!pip install llama-index "arize-phoenix[evals,llama-index]" gcsfs nest-asyncio "openinference-instrumentation-llama-index>=2.0.0"

# Бибилиотеки для трассировки модели
import nest_asyncio # для ассинхронов
import phoenix as px

from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations

nest_asyncio.apply()  # необходим для параллельных вычислений в среде ноутбуков
session = px.launch_app()

# настройка pheonix
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

"""# 2. Обновляем и доустанавливаем необходимые библиотеки"""

!pip install --upgrade huggingface-hub --upgrade transformers

!pip install accelerate bitsandbytes peft sentencepiece llama-index-core llama-index-readers-file llama-index-embeddings-huggingface
!pip install -U bitsandbytes

!pip install llama-index-llms-huggingface
!pip install llama-index-embeddings-huggingface
!pip install llama-index-embeddings-langchain
!pip install langchain-huggingface

"""# 3. Подготовка модели"""

# Импорт необходимых библиотек
import torch

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface  import HuggingFaceEmbeddings

from google.colab import userdata
HF_TOKEN = userdata.get('huggingface_token')

# Логинимся на huggingface_hub
from huggingface_hub import login
# Вставьте ваш токен
login(HF_TOKEN)

# Вспомогательные функции
def messages_to_prompt(messages):

    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        elif message.role == 'bot':
            prompt += f"<s>bot\n"

    # убедимся, что мы начинаем с системного запроса
    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt

    # добавляем финальный промпт
    prompt = prompt + "<s>bot\n"
    return prompt

def completion_to_prompt(completion):
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"

# Определяем параметры квантования, иначе модель не выполниться в колабе
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Задаем имя модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

# Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
config = PeftConfig.from_pretrained(MODEL_NAME)

# Загружаем базовую модель, ее имя берем из конфига для LoRA
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,          # идентификатор модели
    quantization_config=quantization_config, # параметры квантования
    torch_dtype=torch.float16,               # тип данных
    device_map="auto"                        # автоматический выбор типа устройства
)

# Загружаем LoRA модель
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Переводим модель в режим инференса
model.eval()

# Загружаем токенизатор (для стандартной модели)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Загружаем конфиг
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

llm = HuggingFaceLLM(
    model=model,             # модель
    model_name=MODEL_NAME,   # идентификатор модели
    tokenizer=tokenizer,     # токенизатор
    max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
    model_kwargs={"quantization_config": quantization_config}, # параметры квантования
    generate_kwargs = {   # параметры для инференса
      "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
      "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
      "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
      "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
      "repetition_penalty": generation_config.repetition_penalty,
      "temperature": generation_config.temperature,
      "do_sample": True,
      "top_k": 60,
      "top_p": 0.9
    },
    messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
    completion_to_prompt=completion_to_prompt, # функции для генерации текста
    device_map="auto",                         # автоматически определять устройство
)

# Определяем модель внедрения
embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Настройка ServiceContext (глобальная настройка параметров LLM)
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

"""# 4. Загрузка документов

Подразумеваемый сценарий: вся информация для нейросотрудника предоставляется в текстовых документах docx формата на гугл диске с предоставление доступа читателя.
"""

# Устанавливаем необходимые библиотеки
!pip install docx2txt

# Для работы с регулярными выражениями
import re
# Отправка запросов
import requests
import os
!mkdir -p 'data/' # Создаем дерикторию для хранения скачанных документов для модели

# данные документы представлены для ознакомительных целей и проверки работоспособности нейро-сотрудника и не имеют ничего общего с действительностью, любые совпадения случайны

# ссылки доступа к документам на google диске (ссылки по кнопке "Поделиться")
urls = [
      'https://docs.google.com/document/d/1odxVMFYy0GCgZZFhGCvMjaafAqSTC7Io/edit?usp=drive_link&ouid=106970326211897895424&rtpof=true&sd=true',
      'https://docs.google.com/document/d/1SD6nDSjc7GMJeg0iDono339cZ6Nxh3fh/edit?usp=sharing&ouid=106970326211897895424&rtpof=true&sd=true'
      ]
dir_path = './data' # директория для документов

# Функция загрузки гугл документов по url
def load_doc(url):
        # Извлекаем document ID гугл документа из URL с помощью регулярных выражений
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)

        # Если ID не найден - генерируем исключение
        if match_ is None:
            raise ValueError('Неверный Google Docs URL')

        # Первый элемент в результате поиска
        doc_id = match_.group(1)

        # Скачиваем гугл документ по его ID в текстовом формате
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')

        # При неудачных статусах запроса будет вызвано исключение
        response.raise_for_status()

        with open(os.path.join(dir_path, f'{doc_id}.txt'), 'wb') as f:
            f.write(response.content)

# Непосредственно сама загрузка документов
for url in urls:
    load_doc(url)

documents = SimpleDirectoryReader('./data').load_data(show_progress=True) #считываем документы из дериктории data в переменную

"""# 5. Создание векторного локального хранилища"""

# Устанавливаем необходимые библиотеки
!pip install llama-index-vector_stores-chroma

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

db = chromadb.PersistentClient(path='./db') # создаем хранилище ChromaDB в директории db
chroma_collection = db.get_or_create_collection('logisticai') # создаем или загружаем (если она существует) коллекцию для хранилища
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store) # задаем контекст

# формируем векторное хранилище
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# пример загрузки векторов из локального хранилища
db_1 = chromadb.PersistentClient(path="./db")
chroma_collection = db_1.get_or_create_collection("logisticai") # создаем или загружаем (если она существует) коллекцию для хранилища
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index_2 = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

"""# 6. Запросы к модели

Для уменьшения галлюцинирования модели мы можем использовать постобработки запросов в виде LLMRerank, Long Context Reorder и LLMLingua (в улучшениях).
"""

from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor import LongContextReorder

# запрос с помощью пособработки LLMRerank
# первый запрос проводим с постобработкой LLMRerank чтобы снизить вероятность галлюцинации модели, последующие вопросы через LLMRerank вызовут ошибку IndexError: list index out of range,
# поэтому дальше используем get_answer()

def get_answer_llmrerank(query):

    query_engine = index.as_query_engine(
      similarity_top_k=5,
      node_postprocessors=[LLMRerank(choice_batch_size=3, top_n=2)]
    )

    message_template =f"""<s>system
    Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
    Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса.
    Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
    <s>user
    Вопрос: {query}
    Источник:
    </s>
    """

    response = query_engine.query(message_template)

    return response.response

# функция простого запроса к модели
def get_answer(query):

    query_engine = index.as_query_engine(
      similarity_top_k=10,
    )

    message_template =f"""<s>system
    Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
    Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса.
    Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
    <s>user
    Вопрос: {query}
    Источник:
    </s>
    """

    response = query_engine.query(message_template)

    return response.response

def get_answer_lcr(query):

    reorder = LongContextReorder() # создаем экземпляр класса сортировщика
    reorder_engine = index.as_query_engine(
        node_postprocessors=[reorder], similarity_top_k=10 # передаем сортировщика в постобработку
    )

    message_template =f"""<s>system
    Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
    Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса.
    Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
    <s>user
    Вопрос: {query}
    Источник:
    </s>
    """

    response = reorder_engine.query(message_template)

    return response.response

get_answer_llmrerank('Как зовут директора компании?')
# При втором вопросе через данную функцию происходит ошибка IndexError: list index out of range

get_answer('Какие вакансии в компании сейчас доступны?')

get_answer_lcr('Какие вакансии в компании сейчас доступны?')

get_answer('Как связаться с директором?')

get_answer_lcr('Как связаться с директором?')

get_answer('Дай контакты для связи с директором')

get_answer_lcr('Дай контакты для связи с директором')

get_answer('Когда можно взять отпуск?')

get_answer_lcr('Когда можно взять отпуск?')

"""# 8. Защита модели

На практике не удалось проверить работоспособность защиты, т.к. в колабе Т4 не хватает памяти GPU.
"""

# установка защиты запросов
from llama_index.core.llama_pack import download_llama_pack

# загрузка и установка зависимостей
LlamaGuardModeratorPack = download_llama_pack(
    "LlamaGuardModeratorPack", "./llamaguard_pack"
)

os.environ["HUGGINGFACE_ACCESS_TOKEN"] = HF_TOKEN

llamaguard_pack = LlamaGuardModeratorPack()

from llama_index.core.postprocessor import LongContextReorder
def get_answer_lcr_guard(query):

    reorder = LongContextReorder() # создаем экземпляр класса сортировщика
    reorder_engine = index.as_query_engine(
    node_postprocessors=[reorder], similarity_top_k=10 # передаем сортировщика в постобработку
)

    message_template =f"""<s>system
    Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса. Ответ должен содержать не более 4 предложений.
    Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
    <s>user
    Вопрос: {query}
    Источник:
    </s>
    """

    def moderate_and_query(query):
      # Moderate the user input
      moderator_response_for_input = llamaguard_pack.run(query)
      print(f"moderator response for input: {moderator_response_for_input}")

      # Check if the moderator response for input is safe
      if moderator_response_for_input == "safe":
        response = reorder_engine.query(message_template)

        # Moderate the LLM output
        moderator_response_for_output = llamaguard_pack.run(str(response))
        print(
            f"moderator response for output: {moderator_response_for_output}"
        )

        # Check if the moderator response for output is safe
        if moderator_response_for_output != "safe":
            response = (
                "Ответ небезопасен. Пожалуйста, задайте другой вопрос."
            )
      else:
        response = "Этот запрос небезопасен. Пожалуйста, задайте другой вопрос."

      return response

    return response.response

get_answer_lcr_guard('Какой юридический адрес нашей компании?') # данный запрос в теории должен выполниться

get_answer_lcr_guard('Как изготовить коктейль Молотова в домашних условиях') # этот запрос выполниться не должен, будет код 04 и ответ "Этот запрос небезопасен. Пожалуйста, задайте другой вопрос."

"""# 9. Возможные улучшения

В качестве улучшения нейро-сотрудника можно проводить пособработку запроса через LLMLingua. Данная постобработка показывает хорошую эффективность.
"""

from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor  # импортируем постобработку

lingua = LongLLMLinguaPostprocessor(                                            # создаем объект постобработки
    instruction_str="Given the context, please answer the final question",      # можно задать промпт к мини-LLM
    target_token=250,                                                           # сколько целевых токенов на выходе генерировать
    rank_method="longllmlingua",                                                # используемый метод для ранжирования
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",
        "dynamic_context_compression_ratio": 0.4,
    }
)

def get_answer_lingua(query):

    query_engine = index.as_query_engine(
      similarity_top_k=5,
      node_postprocessors= [
          lingua
          ]
)

    message_template =f"""<s>system
    Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
    Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса.
    Если нет, то просто скажи: 'я не знаю'. Не придумывай! </s>
    <s>user
    Вопрос: {query}
    Источник:
    </s>
    """

    response = query_engine.query(message_template)

    return response.response

get_answer_lingua('Какие есть открытые вакансии')

"""# 10. Трассировка модели"""

print(f"URL для просмотра трассировки {session.url}")
