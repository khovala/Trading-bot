# DVC пайплайны и трекинг экспериментов с MLFlow

## Работа с DVC

[Документация](https://dvc.org) DVC

[Обучающий курс](https://learn.dvc.ai) по DVC

### Настройка

1. Устанавливаем DVC

```bash
pip install dvc
```

2. В папке проекта инициализируем DVC проект

```bash
dvc init

# Если папка проекта не является репозиторием, то
dvc init --no-scm
# ⚠️ Обратите внимание что тогда будут недоступны фукнции 
# версионирования данных. Только пайплайны и метрики
```

3. Настраиваем DVC

```bash
# отключаем сбор аналитики
dvc config core.analytics false
# настраиваем коннект к S3 (опционально, если не версионируете данные)
# флаг --local записывает настройки в .dvc/config.local,
# который добавлен в .gitignore. Здесь можно хранить секреты
dvc remote add --local -d <connection_name> s3://<bucket_name>
dvc remote modify --local <connection_name> endpointurl http://<url>
dvc remote modify --local <connection_name> access_key_id <aws_key_id>
dvc remote modify <connection_name> secret_access_key <aws_token>

# Например
dvc remote add --local -d my_s3 s3://dvc
dvc remote modify --local my_s3 endpointurl http://172.11.1.11:9000
dvc remote modify --local my_s3 access_key_id fsdfdafsf
dvc remote modify --local my_s3 secret_access_key fdsgfewgfwfd
```

4. (опционально, если не версионируете данные) Добавляем в трекинг гита
```bash
git add .dvc/config
```

5. (опционально, если не версионируете данные) Коммитим изменения
```bash
git commit -m 'configure dvc'
```

6. (опционально, если не версионируете данные) Пушим в репозиторий
```bash
git push
```
### Пайплайны

[Инструкция](https://dvc.org/doc/start/data-pipelines/data-pipelines) по пайплайнам

Пайплайн можно создать через CLI (`dvc stage add ...`), либо с помощью файла `dvc.yaml`

[Пример](./dvc.yaml) `dvc.yaml`

Посмотреть граф пайплайна
```bash
dvc dag
```

Выполнить весь пайплайн
```bash
dvc repro

# Если необходим запуск всех стадий, даже если по ним DVC не видит изменений
dvc repro --force
```

Запуск пайплайна до определённой стадии
```bash
dvc repro <stage_name>

# Либо
dvc repro <stage_name> --force
```
### Метрики и параметры

Если в пайплайне задали метрики, то можно посмотреть текущие метрики
```bash
dvc metrics show
```

Если работаете в гит репозитории, то можно сравнить текущие метрики с последним коммитом
```bash
dvc metrics diff
```

Если в пайплайне есть раздел `params`, то можно сравнить текущие параметры с последним коммитом
```bash
dvc params diff
```

### Версионирование данных

Чтобы начать версионировать данные необходимо добавить их в отслеживание DVC

1. Отслеживаем файл
```bash
# Можно добавить как один файл, так и директорию
# В таком случае, если в директории изменится хоть один файл,
# то DVC будет считать всю эту директорию новой версией
# и затащит в S3 полностью
dvc add <filename>
```

2. Добавить новый файл `<filename>.dvc` с MD5 оригинального файла в трекинг гита
```bash
git add <filename>.dvc
```

3. Коммитим и пушим в гит
```bash
git commit -m 'add dataset to dvc'
git push
```

4. Пушим оригиналный файл в S3
```bash
dvc push
```

DVC позволяет переключаться между версиями файлов когда переключаетесь между коммитами гита

Получить версии файлов с текущего коммита
```bash
dvc checkout
```

### Доп фичи

Ещё у DVC есть:
- [графики](https://dvc.org/doc/start/data-pipelines/metrics-parameters-plots#collecting-metrics-and-plots)
- [отслеживание экспериментов](https://dvc.org/doc/start/experiments/experiment-tracking)
- [реестр моделей](https://dvc.org/doc/start/model-registry)
- [плагин для VSCode](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc)

## Работа с MLFlow

[Пример docker compose с MLFlow, Minio S3 и PostgreSQL](../mlflow_docker)

По умолчанию MLFlow доступен по 5000 порту. В браузере перейти по адресу `http://<mlflow_IP>:5000`

Пример использования MLFlow в [train.py](./src/models/train.py) и [evaluate.py](./src/models/evaluate.py)

**MLFlow** позволяет отслеживать эксперименты, сохранять артефакты и модели, управлять жизненным циклом моделей и деплоить модели

**Эксперименты** в MLflow представляют собой ключевой элемент для организации и отслеживания процессов обучения моделей машинного обучения. Они позволяют систематизировать и группировать различные **запуски (runs)**, что облегчает анализ результатов и управление жизненным циклом моделей.

Чтобы MLFLow мог отслеживать эксперименты необходимо:
1. Установить библиотеку для python
```shell
pip install mlflow
```

2. В коде указать адрес MLFlow
```python
import mlflow

mlflow.set_tracking_uri(f'http://{mlflow_host}:{mlflow_port}')

# Например
mlflow.set_tracking_uri(f'http://172.10.10.10:5000')
```

3. Указать имя эксперимента
```python
mlflow.set_experiment('news_sentiment')
```

4. Теперь можно логировать всю необходимую информацию

У MLFLow есть несколько способов логирования:

1. Автоматическое
```python
import mlflow

mlflow.<module_name>.autolog

# Например
mlflow.sklearn.autolog
mlflow.keras.autolog
mlflow.pytorch.autolog
```
Полный перечень поддерживаемых библиотек и фреймворков в [описании Python API](https://mlflow.org/docs/latest/python_api/index.html)

2. Ручное логирование - выбираем сами что логировать (метрики, модели, параметры, артефакты и т. п.)
```python
import mlflow

# Логируем только один параметр
mlflow.log_param('learning_rate', 0.01)

# Логируем группу параметров
params = {'learning_rate': 0.01, 'n_estimators': 10}
mlflow.log_params(params)

# Логируем одну метрику
mlflow.log_metric('mse', 234.4)

# Логируем группу метрик
metrics = {"mse": 234.41, "rmse": 50.00}
mlflow.log_metrics(metrics)

# Логируем модель
mlflow.<module_name>.log_model(...)
# Например
mlflow.sklearn.log_model(
    sk_model=classifier, # Непосредственно сама модель
    artifact_path="logreg", # Её путь в артифактах
    input_example=X[:1]  # Пример данных для предсказания
)

# Логировать артефакты (датасеты, промежуточные файлы и т. п.)
mlflow.log_artifact(
    local_path=<путь к артефакту>,
    artifact_path=<путь в артефактах MLFlow>,  # опционально
    run_id=<id_запуска_эксперимента>  # опционально
    )
# Например, загрузим тренировочный датасет в папку datasets в MLFlow артефактах
mlflow.log_artifact('./data/train_dataset.csv', 'datasets')
```

**Артефакт** в контексте MLflow — это любой файл, который создается и фиксируется в ходе выполнения эксперимента или запуска модели машинного обучения. Артефакты могут включать в себя различные типы данных, такие как:
- **Модели**: Сохраненные веса и параметры моделей
- **Конфигурационные файлы**: Файлы, содержащие настройки и параметры
- **Результаты**: Графики, отчеты и т. п.
- **и т. д.**

***

Если эксперимент разбит на несколько скриптов, то необходимо передавать между ними `run_id` запуска эксперимента

`script_1.py`
```python
import mlflow

mlflow.set_tracking_uri(f'http://172.10.10.10:5000')
mlflow.set_experiment('my_experiment')

with mlflow.start_run() as run:
    ...  # весь необходимый код
    print(run.info.run)  # получить ID текущего запуска
```

`script_2.py`
```python
import mlflow

mlflow.set_tracking_uri(f'http://172.10.10.10:5000')
mlflow.set_experiment('my_experiment')

# Указываем run_id из предыдущего скрипта
with mlflow.start_run(run_id=run_id) as run:
    ...  # весь необходимый код
```

## Код-стайл

```bash
uv run ruff format .
uv run ruff check . --fix
```

## Настройка окружения

С помощью [UV](https://docs.astral.sh/uv/)
```bash
uv sync
```

С помощью pip
```bash
pip install -r requirements.txt
```