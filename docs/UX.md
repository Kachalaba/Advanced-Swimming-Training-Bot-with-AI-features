# Sprint Bot Scenario Playbook

## Web Product: Dryland Analysis
- **Entry point**: `/dryland` opens the dark SPRINT AI sport landing page with real athlete history, not demo metrics.
- **Exercise selection**: athlete/clinician must choose `Squat`, `Lunge`, or `Push-up` before upload. The selected profile controls the backend metric target and avoids guessing the exercise from noisy pose data.
- **Capture guidance**: fixed side view, full body visible, no cropped joints, complete ready -> effort -> ready repetitions.
- **Quality gate**: clips with too few metric-ready frames or low pose coverage are rejected with reshoot guidance instead of producing a weak score.
- **Result page**: `/dryland/{jobId}` shows annotated evidence video, confirmed reps, tempo, ROM, stability, pose coverage, metric-ready frames, and a per-rep table.
- **History**: successful jobs can be saved to an athlete and then appear in the dryland sport overview.

## /start Onboarding
- **Happy path**: роль → приватность → имя → тренер → группа → язык → карточка профиля.
- **Защита**: отказ приватности сбрасывает состояние и отзывает инвайт; неверные trainer-ID/инвайты дают подсказки.
- **Сценарные тесты**: `tests/test_onboarding_flow.py` покрывает happy-path, отказ приватности и deep-link-инвайт.

```mermaid
flowchart LR
    start[/Команда /start/] --> role{Выбор роли}
    role -->|Тренер| privacy{Поделиться данными?}
    role -->|Атлет| privacy
    privacy -->|Нет| abort[/Отменяем онбординг/]
    abort --> exit_state[(FSM reset)]
    privacy -->|Да| profile["Запрос имени и контакта"]
    profile --> trainer[Проверка trainer ID или инвайта]
    trainer --> group[Выбор группы]
    group --> locale[Выбор языка]
    locale --> card[Показываем карточку профиля]
    card --> done[/Онбординг завершён/]
```

## /help Справка
- Сообщение разбито на блоки: ввод, история, сравнение, рекорды, лидерборд, экспорт.
- Строки локализованы (uk/ru) и проверяются в `tests/test_bot_i18n.py`.
- Хэндлер не требует состояний — доступен всегда, безопасен к спаму.

```mermaid
sequenceDiagram
    participant U as Пользователь
    participant B as Бот
    U->>B: /help
    B-->>U: Заголовок + интро
    B-->>U: Блок "Ввод результатов"
    B-->>U: Блок "История"
    B-->>U: Блок "Сравнение"
    B-->>U: Блок "Рекорды"
    B-->>U: Блок "Лидерборд"
    B-->>U: Блок "Экспорт"
```

## Мастер ввода сплитов
- Шаги: стиль/дистанция → шаблон → сплиты → тотал → подтверждение.
- Поддерживает форматы `mm:ss.ss`, `см/мс`, `repeat/cancel`, автосумму и выравнивание.
- Тесты (`tests/test_add_wizard.py`, `tests/test_add_wizard_i18n.py`) закрывают happy-path, отмену, повтор и ошибки формата.

```mermaid
stateDiagram-v2
    [*] --> preset
    preset: Стиль/дистанция/шаблон
    preset --> splits
    splits: Ввод сплитов
    splits --> splits: Подсказка формата / повтор шага
    splits --> total: Тотал и валидация
    total --> confirm: Подтверждение сохранения
    total --> splits: Ошибка формата → возврат
    confirm --> [*]
    [*] --> cancel: /cancel в любой момент
    cancel --> [*]
```
