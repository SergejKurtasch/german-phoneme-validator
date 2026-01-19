# Инструкция по загрузке моделей на Hugging Face Hub

## Шаг 1: Установка зависимостей

Убедитесь, что установлен `huggingface_hub`:

```bash
pip install huggingface_hub
```

## Шаг 2: Авторизация в Hugging Face

Выполните одну из команд для авторизации:

```bash
# Вариант 1: Интерактивный вход
huggingface-cli login

# Вариант 2: Использование токена напрямую (будет запрошен в скрипте)
# Токен можно получить на https://huggingface.co/settings/tokens
```

## Шаг 3: Создание репозитория на Hugging Face

1. Перейдите на https://huggingface.co/new
2. Создайте новый репозиторий с типом **Model**
3. Имя репозитория (например): `german-phoneme-models`
4. Видимость: **Public** (для публичного доступа) или **Private** (для приватного)
5. Нажмите **Create repository**

**Важно:** Запомните полное имя репозитория в формате `username/repo-name`
   Например: `SergejKurtasch/german-phoneme-models`

## Шаг 4: Загрузка моделей

Запустите скрипт загрузки из корня проекта:

```bash
python tools/upload_models.py --repo-id SergejKurtasch/german-phoneme-models --artifacts-dir artifacts/
```

### Параметры скрипта:

- `--repo-id` (обязательно): Полное имя репозитория на HF Hub
  - Формат: `username/repo-name`
  - Пример: `SergejKurtasch/german-phoneme-models`

- `--artifacts-dir` (опционально): Путь к папке с моделями
  - По умолчанию: `artifacts/`
  - Указывайте, если модели находятся в другом месте

- `--token` (опционально): Hugging Face токен
  - Если не указан, используется кэшированный токен (после `huggingface-cli login`)
  - Можно получить на https://huggingface.co/settings/tokens

- `--repo-type` (опционально): Тип репозитория
  - По умолчанию: `model`
  - Возможные значения: `model`, `dataset`, `space`

### Пример полной команды с токеном:

```bash
python tools/upload_models.py \
  --repo-id SergejKurtasch/german-phoneme-models \
  --artifacts-dir artifacts/ \
  --token YOUR_HF_TOKEN_HERE
```

## Шаг 5: Проверка загрузки

После успешной загрузки:

1. Перейдите на страницу репозитория: `https://huggingface.co/SergejKurtasch/german-phoneme-models`
2. Убедитесь, что все папки моделей загружены (например, `b-p_model/`, `a-E_model/`, и т.д.)
3. Проверьте, что в каждой папке есть все файлы:
   - `best_model.pt`
   - `config.json`
   - `feature_cols.json`
   - `feature_scaler.joblib`

## Шаг 6: Обновление DEFAULT_REPO_ID в коде (если нужно)

Если имя репозитория отличается от `SergejKurtasch/german-phoneme-models`, 
обновите константу в `core/downloader.py`:

```python
# В файле core/downloader.py
DEFAULT_REPO_ID = "ваш-username/ваш-repo-name"
```

## Важные замечания

1. **Проверка имен папок:** Скрипт автоматически проверит имена папок на наличие 
   Windows-небезопасных символов (`:`, `\`, `/`, и т.д.) и предупредит об этом.

2. **Первый запуск:** Первая загрузка может занять время, так как загружаются все 
   22 модели (~несколько десятков МБ).

3. **Обновления моделей:** При последующих запусках скрипта будут загружены только 
   измененные файлы (благодаря механизму ETag в Hugging Face Hub).

4. **Структура в HF Hub:** Структура в репозитории должна совпадать с локальной:
   ```
   german-phoneme-models/
   ├── b-p_model/
   │   ├── best_model.pt
   │   ├── config.json
   │   ├── feature_cols.json
   │   └── feature_scaler.joblib
   ├── a-E_model/
   │   └── ...
   └── ...
   ```

## Тестирование загрузки

После загрузки можно проверить, что модели доступны:

```python
from german_phoneme_validator.core.downloader import get_model_assets

# Попробуйте загрузить модель
model_dir = get_model_assets('b-p')
print(f"Model downloaded to: {model_dir}")
print(f"Model files: {list(model_dir.glob('*'))}")
```

## Решение проблем

### Ошибка авторизации
- Убедитесь, что выполнили `huggingface-cli login` или указали корректный токен
- Проверьте, что токен имеет права на запись в репозиторий

### Ошибка "Repository not found"
- Проверьте, что репозиторий создан и имя указано правильно
- Убедитесь, что у вас есть права на запись в репозиторий

### Медленная загрузка
- Это нормально для первой загрузки всех моделей
- Проверьте скорость интернет-соединения

### Предупреждения о спецсимволах
- Если скрипт предупреждает о Windows-небезопасных символах в именах папок,
  проверьте их. Обычно это не проблема, так как имена папок уже нормализованы.
