# bug_reporter_bot
Bot for finding known bugs in vector database

# 🤖 SoftInterMob — Bug-reporter Bot

Телеграм-бот, который отвечает на вопросы об известных багах в играх.  
Стек: **Pinecone Serverless llama-text-embed-v2** + **n8n** + **Google Sheets**.

| Слой                     | Технология                                   |
| ------------------------ | -------------------------------------------- |
| Интерфейс                | Telegram                                     |
| Поиск (R из RAG)         | Pinecone Serverless `llama-text-embed-v2`, cosine |
| Генерация (G из RAG)     | OpenAI `gpt-4o-mini` (опционально)           |
| Оркестрация              | n8n (cloud/self-host)                        |
| Логи и метрики           | Google Sheets через Service Account          |
| Мониторинг Pinecone      | n8n -> Google Sheets                         |

---

## 1️⃣ Быстрый старт

```bash
git clone https://github.com/visllove/bug_reporter_bot.git
cd softintermob-bug-bot

# 1. Зависимости Python
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Secret keys
cp .env                

# 3. Загрузка текста багов в Pinecone
python load_bug_info_to_db.py       # выполняется один раз для загрузки текста

# 4. Импортируйте workflow-файлы в n8n (Settings → Import),
#    привяжите credentials (Telegram, Pinecone, OpenAI, Sheets), активируйте.

# 5. Напишите боту в Telegram:
#    «зависает при загрузке уровня» — получите описание бага.
```

## 2️⃣ Задайте переменные окружения (.env)

**Pinecone**
PINECONE_API_KEY=
PINECONE_ENV=
PINECONE_HOST=
PINECONE_INDEX=

**Telegram**
TELEGRAM_BOT_TOKEN=

**Google Sheets (Service Account)**
GOOGLE_SERVICE_ACCOUNT_JSON=

**OpenAI - опционально, только если нужны отдельные эмбеддинги и ответы, сгенерированные LLM**
OPENAI_API_KEY=

## 3️⃣ Service Account для Google Sheets

**Важно создавать именно Service account, не делать аутентификацию через OAuth2, так как нам важен длительный доступ и автономная работа, для которой подходит сервисный аккаунт.**

1. Google Cloud Console → IAM & Admin → Service Accounts → Create.

2. Дайте роль “Editor → Google Sheets API Service Agent”. Важно добавить нужные API (Google Drive и Google Sheets) к этому аккаунту.

3. Keys → Add key → JSON — скачайте gcp-sa-softintermob.json.

4. Откройте нужную таблицу в Google Sheets → Share → добавьте сервис-аккаунт как Editor.

5. В Credentials n8n выберите Google Sheets → Service Account.


## 4️⃣ Создание индекса Pinecone

Создайте аккаунт на https://app.pinecone.io - сервисе для управления БД Pinecone. Далее необходимо создать индексы для хранения векторных представлений текстовых данных

**Эмбеддингом, то есть преобразователем текста в векторы, в данном случае выступает встроенная в функционал Pinecone модель llama-text-embed-v2, которая хорошо работает с различными языками, в том числе с русским.**

## 5️⃣ Особенности проекта

Показатель topK равен 1, это значит, что будет выбираться всегда только один наиболее релевантный запросу баг. Если использовать отдельную LLM для обработки запросов, то можно ставить topK > 1.

Отдельной LLM для обработки ответов по умолчанию нет, так как используется логика с порогом по метрике cosine (score установлен на 0.3). Если самый подходящий ответ имеет векторное расстояние, большее порога -> бот отвечает описанием бага, если меньшее - бот отвечает "Информации о баге не найдено, попробуйте описать запрос подробнее".

Логирование в Google Sheets включает метрику cosine (score), соответственно, можно тестировать ответы бота по ним. Другой вариант отладки - проверка результатов прямо в интерфейсе Pinecone, там тоже есть возможность проверить любые запросы и посмотреть, насколько сильно topK возможных ответов близки к запросу.





