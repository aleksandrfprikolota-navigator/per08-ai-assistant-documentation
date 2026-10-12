# PEr08. RAG-ассистент по документации «1С-Премиум»

Консольный ассистент отвечает на вопросы по материалам компании [«1С-Премиум»](https://www.1c-prem.ru/about/): страница о компании, контакты, график работы, описания услуг и статьи. Один и тот же RAG-пайплайн подключён к двум моделям: OpenAI `gpt-4o-mini` и GigaChat `GigaChat-2-Max`.

Репозиторий: [per08-ai-assistant-documentation](https://github.com/aleksandrfprikolota-navigator/per08-ai-assistant-documentation).

## Как устроено

1. Тексты из `data/*.txt` режутся на чанки около 500 символов с перекрытием 100 символов. Длинная страница сайта сначала делится по абзацам и строкам, затем по предложениям.
2. Чанки переводятся в векторы моделью OpenAI `text-embedding-3-small` и кладутся в ChromaDB вместе с именем исходного файла.
3. Вопрос ищется в базе, в модель уходят 3 ближайших фрагмента.
4. Повтор того же вопроса берётся из кэша SQLite.

Папки:

| Папка | Что делает |
|---|---|
| `5-8-final/assistant_api` | ответы через OpenAI и оценка RAGAS |
| `5-8-final/assistant_giga` | ответы через GigaChat |
| `5-8-final/assistant_*/data` | 24 текста базы знаний |

Embeddings GigaChat на используемом ключе возвращают `402 Payment Required`, поэтому индекс в обеих версиях строит OpenAI. Генерация ответов во второй версии идёт через GigaChat. Если выбранная модель Сбера отвечает `500`, клиент пробует `GigaChat-2-Max`, затем `GigaChat-2-Pro`.

## Запуск

Нужен Python 3.11. Ключи лежат в `5-8-final/.env` и в репозиторий не входят. Образец имён переменных: `5-8-final/.env.example`.

```powershell
cd D:\PythonProject_PI\5\PEr08_AI_AssistantDocumentation\5-8-final
py -3.11 -m venv venv_py311
.\venv_py311\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

В `.env` должны быть `OPENAI_API_KEY`, `GIGACHAT_AUTH_KEY` и `GIGACHAT_RQUID`.

Перед проверкой включите UTF-8 в текущем окне PowerShell, иначе рамка консольного приложения может не напечататься:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

Первый запуск строит индекс: 24 файла, 947 чанков. Повторный запуск индекс не пересобирает. Чтобы заново проиндексировать документы, удалите папку `chroma_db` в каталоге ассистента.

## Проверка OpenAI

```powershell
cd D:\PythonProject_PI\5\PEr08_AI_AssistantDocumentation\5-8-final\assistant_api
..\venv_py311\Scripts\python.exe -X utf8 cache.py
..\venv_py311\Scripts\python.exe -X utf8 vector_store.py
..\venv_py311\Scripts\python.exe -X utf8 rag_pipeline.py
..\venv_py311\Scripts\python.exe -X utf8 evaluate_ragas.py
..\venv_py311\Scripts\python.exe -X utf8 app.py
```

`app.py` — живой диалог. Команды: `stats`, `clear`, `exit`.

`rag_pipeline.py` задаёт четыре вопроса по компании и повторяет первый, чтобы показать кэш. На проверке 12 октября 2026 года ответы совпали с документами: «1С-Премиум» — дочерняя компания фирмы «1С»; офис работает пн–пт с 10:00 до 19:00, обед с 13:00 до 14:00; почта персонального предложения `corp@1c-prem.ru`, ИНН `7713493104`; «1С:Шина» нужна для обмена между системами. Повторный вопрос пришёл из кэша.

RAGAS по четырём вопросам с эталонными ответами: Faithfulness `0.7917`, Context Precision `0.8125`, средний балл `0.8021`.

## Проверка GigaChat

```powershell
cd D:\PythonProject_PI\5\PEr08_AI_AssistantDocumentation\5-8-final\assistant_giga
..\venv_py311\Scripts\python.exe -X utf8 cache.py
..\venv_py311\Scripts\python.exe -X utf8 gigachat_client.py
..\venv_py311\Scripts\python.exe -X utf8 vector_store.py
..\venv_py311\Scripts\python.exe -X utf8 rag_pipeline.py
..\venv_py311\Scripts\python.exe -X utf8 app.py
```

На той же проверке GigaChat-2-Max ответил по тем же фактам и отдельно подтвердил, что визит в офис только по предварительной записи не менее чем за один рабочий день (`faq_grafik_raboty.txt`). Сообщение про `402` у embeddings ожидаемо: после него индекс строится через OpenAI, а текст ответа пишет GigaChat.
