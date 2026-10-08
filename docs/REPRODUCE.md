# Воспроизведение

Для основной модели начните с [раздела 19](#19-сетевая-типология-и-проверка-аналогов), для сборки атласа с [раздела 20](#20-переотрисовка-атласа-из-сохранённых-данных). Разделы 3–18 описывают архивные расчёты v1.1.

Python 3.10. Команды выполняются из корня репозитория после активации виртуального окружения. Версии закреплены в `requirements*.txt`. Статусы фактических проверок — в [протоколе выпуска](../reports/dmon-plateau-2026-09-25/verification.json).

## 1. Подготовка данных

```sh
python -m pip install -r requirements-models.txt -r requirements-visuals.txt
python scripts/download_sources.py
python scripts/prepare.py
python scripts/prepare_graphs.py
```

Для RAR нужен `bsdtar`, `unrar`, `7z` либо Windows `tar`; пакет Debian/Ubuntu — `libarchive-tools`. Загрузчик проверяет SHA-256 трёх исходных архивов. Контрольная панель имеет SHA-256 `a175b7646a528e900d61dd4ade8d84fb4549ffcbb977d845340107b4f3b3c0fa`, 2 016 МО × 24 месяца. При несовпадении хеша новая выгрузка не считается тем же набором.

## 2. Точная сборка опубликованного атласа

```sh
python scripts/build_review_extension.py
python scripts/build_research_atlas.py --reference-run reports/experiments/2026-09-23-v2 --validation-run reports/experiments/2026-09-23-v2/validation --contest-run reports/review-2026-09-24/atlas_extension.json --map-data reports/contest-v3/municipal_map.json --v12 reports/v1.2 --split-assets --output artifacts/atlas-preview/index.html
```

Сборка использует полные опубликованные метки и научные результаты, ничего не переобучает. Для побайтовой проверки сравните полученный HTML с `docs/index.html`, а файл из соседней папки `assets` с одноимённым файлом в `docs/assets` (`cmp` в Linux или SHA-256). Открывайте `index.html` обычным современным браузером, сохранив рядом папку `assets`; сеть не нужна.

Чтобы получить один автономный HTML со всеми данными внутри, уберите `--split-assets`. В обоих вариантах gzip-пакет распаковывается стандартным браузерным `DecompressionStream`; `--uncompressed` позволяет собрать несжатую версию. Геометрия карты и численные результаты при упаковке не меняются.

Независимое повторение производных таблиц и карты:

```sh
python scripts/evaluate_published.py
python scripts/describe_frozen.py
python scripts/build_map.py --gpkg artifacts/sources/acquisition/t_dict_municipal_districts_poly.gpkg --output artifacts/rebuilt-map.json
```

Сверяйте значения, конвенции и охват с опубликованными артефактами. Внешняя геометрия не входит в признаки моделей.

## 3. Архив v1.1: точное переобучение KMeans4

```sh
python scripts/run_research.py --config configs/reproduce_kmeans4.json --execute-clustering
python scripts/run_research.py --compare-run runs/research-<напечатанный-идентификатор>
```

Первая команда обучает только годовой KMeans4 с исходными параметрами: seed=1729, n_init=20. Вторая сверяет все 2 016 идентификаторов и метки с опубликованными. Она сообщает буквальное совпадение и эквивалентность с перестановкой номеров групп; ненулевой код означает отличие буквальных меток. Новый каталог содержит `status.json`, `metrics.json`, `labels/` и provenance с хешами модулей.

## 4. Полная историческая серия

```sh
python scripts/run_research.py --config configs/research_annual.json --execute-clustering
python scripts/run_research.py --config configs/research_months.json --execute-clustering
python scripts/run_research.py --config configs/research_stability.json --execute-clustering
python scripts/run_research.py --config configs/research_validation.json --execute-clustering
```

Годовая и месячная стадии создают 26 и 78 разбиений; стадия устойчивости — 56 возмущений; валидация повторяет зафиксированный протокол 2024 года. Каждый запуск печатает новый `runs/research-<timestamp>`. Для экспорта передайте эти четыре пути в `scripts/export_research_results.py` через `--annual`, `--months`, `--stability`, `--validation` и новый `--output`. Архивные численные результаты не перезаписываются; происхождение старых метрик сохраняется отдельно от расширенных конвенций.

## 5. Совместная и временная модели

```sh
python scripts/run_contest.py --config configs/followup_small_k.json --execute-clustering
python scripts/run_contest.py --config configs/followup_joint.json --execute-clustering
python scripts/run_contest.py --config configs/followup_stability.json --execute-clustering
python scripts/run_contest.py --config configs/followup_temporal_pilot.json --execute-clustering
python scripts/run_contest.py --config configs/followup_temporal.json --execute-clustering
```

Порядок первых двух обязателен: joint переиспользует рассчитанные KMeans2/3. Стадия stability использует кэш дорожного графа из joint, сверяет хеши и делает восемь дополнительных запусков с другими начальными разбиениями и порядком обхода. Пилот проверяет три месяца, полный временной запуск — 24. Параметры результатов находятся в `followup.output`; существующие каталоги не перезаписываются. Для повторного запуска сохраните копии конфигов с новыми выходными путями и соответствующим `baseline_dir` у joint и путями `joint_dir` / `baseline_dir` у stability.

Экспорт принимает только завершённые результаты:

```sh
python scripts/export_followup.py --run runs/review-20260924-small-k --run runs/review-20260924-joint --run runs/review-20260924-joint-stability --run runs/review-20260924-temporal-pilot --run runs/review-20260924-temporal --output reports/review-repeated
```

Выход: метки всех МО, таблицы ICVI, трассы целевой функции, конфиги, научные снимки реализации и manifest SHA-256. Это разведочные расчёты; уже изученный 2024 год не становится новым отложенным тестом.

## 6. Внешняя проверка и иллюстрации

```sh
python scripts/external_validation_v4.py --output artifacts/external-repeated --bootstrap 500
python scripts/build_review_figures.py --external-results artifacts/external-repeated/results.json --output-dir artifacts/figures-repeated
```

Внешний контроль использует фиксированные исходные метки, регион, тип МО, расходы и индекс доступности рынков. Он не обучает новую кластеризацию. Выход должен быть пустым или новым. Для переноса пересчитанного внешнего контроля в атлас укажите `--external-results artifacts/external-repeated/results.json` у `build_review_extension.py`. Без этого флага сборка использует опубликованный снимок `reports/external-v4/results.json`.

## 7. Тесты и браузер

```sh
python -m unittest discover -s tests -v
python -m compileall -q sbercluster scripts
npm install --no-save --package-lock=false playwright@1.62.1
npx playwright install chromium
node tests/browser/map_regressions.cjs docs/index.html chromium
```

В чистом клоне проверки локальной панели пропускаются до её подготовки. В браузерном тесте используются настоящие щелчки, перетаскивания, касания и клавиатура; проверяются масштаб, подсказки, переключение совместных моделей и помесячная сеть. Вместо `chromium` можно указать установленный `chrome` или `msedge`.

## 8. Переносимость и Docker

```sh
docker build -t economic-neighbors .
docker run --rm economic-neighbors python -m unittest discover -s tests -v
```

Для обучения смонтируйте подготовленную `data/processed` в `/app/data/processed`, а выходной `runs` — в `/app/runs`. Для дорожных экспериментов также нужны `artifacts/sources/acquisition`. `.git` и локальные рабочие материалы в образ не включаются; происхождение сохраняет хеши кода, а необязательный build argument `SOURCE_REVISION` — идентификатор исходного коммита.

GitHub Actions собирает контейнер, обучает основной KMeans4 на Linux, сверяет метки и выполняет малый пилот. Фактический статус конкретного выпуска указан в протоколе проверки, а не предполагается по наличию workflow.

Переносимые команды ограничивают число вычислительных потоков. Необязательный [Windows supervisor](RESOURCE_LIMITS.md) включается через `--supervisor PROFILE`; его машинные настройки хранятся отдельно от научного конфига.


## 9. Повторная проверка: регион, DMoN и сетка временных связей

Эта серия использует уже опубликованные исходные разбиения и дорожную матрицу из `reports/graphs`. Загрузчик проверяет manifest, порядок всех 2 016 ID, признаки, параметры и SHA-256. Старые модели повторно не обучаются.

После подготовки панели из раздела 1:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-dmon.txt
python -m unittest discover -s tests -p test_dmon.py -v
python -m unittest discover -s tests -p test_round2.py -v
python scripts/run_contest.py --config configs/round2_region_control.json --execute-clustering
python scripts/run_contest.py --config configs/round2_dmon_pilot.json --execute-clustering
python scripts/run_contest.py --config configs/round2_temporal_pilot.json --execute-clustering
```

Пилоты: DMoN K=2, seed 1729, 200 эпох; временная модель — три месяца, ω=0,03 и три seed. Масштабирование в `round2.yml` разрешено только после успешных тестов и двух пилотов, каждый не длиннее 120 секунд на используемой машине. Этот предел — бюджет пилота, не обещание времени на другом компьютере.

```sh
python scripts/run_contest.py --config configs/round2_dmon.json --execute-clustering
python scripts/run_contest.py --config configs/round2_temporal_grid.json --execute-clustering
python scripts/export_round2.py --run runs/round2-20260924-region_control
python scripts/export_round2.py --run runs/round2-20260924-dmon
python scripts/export_round2.py --run runs/round2-20260924-temporal_grid
python scripts/build_round2_figures.py
```

DMoN: K=2/4, seeds 1729/2718/3141, по 1 000 эпох; ни seed, ни checkpoint не выбираются по внешним показателям. Временная серия: ω=0/0,01/0,03/0,06/0,1, три seed, 24 месяца; два прежних результата используются повторно, новые 13 рассчитываются. Четыре итерации Leiden — фиксированный бюджет, не заявление о сходимости. Снимки исходного кода хранятся по содержимому в общем каталоге `source_objects`; одинаковые файлы не дублируются между новыми запусками.

При повторном выполнении меняйте `followup.output` в копии конфига и каталог экспорта: существующие результаты защищены от перезаписи. Workflow использует `scripts/run_round2_stage.py`: завершённый опубликованный этап пропускается только после проверки хешей кода, конфигурации и файлов; незавершённые сырые результаты сохраняются отдельным CI-артефактом. Это позволяет продолжить серию без повторения готовых этапов.

Для просмотра пересчитанных чисел достаточно JSON/CSV; опубликованный атлас использует проверенные архивы `reports/round2-2026-09-24`.

GitHub Actions `Round two experiments` выполняет эту последовательность только при ручном запуске. Обычные изменения описания не запускают обучение повторно. Основной `Python checks` продолжает проверять каждое изменение научного кода, JSON и HTML.

Новый региональный источник Росстата готовится отдельно, без обучения:

```sh
python scripts/download_external_round2.py
python scripts/prepare_external_round2.py
```

Правила сопоставления и географический охват — в `reports/external-round2/README.md`. Зарплата работников организаций не подменяет доходы населения.


## 10. DMoN с остановкой по плато

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-dmon.txt
python -m unittest discover -s tests -p test_dmon.py -v
python scripts/run_contest.py --config configs/dmon_plateau.json --execute-clustering
python scripts/export_round2.py --run runs/dmon-plateau-20260925 --output reports/dmon-plateau-2026-09-25 --compact-traces
python scripts/build_dmon_plateau.py
python scripts/build_review_extension.py
```

Используйте новый выходной каталог для повторения: экспорт не перезаписывает опубликованные свидетельства. Правило остановки задано в конфиге до обучения: после 3 000 эпох, каждые 500, три последовательных окна с улучшением лучшего loss не более 0,0001 и сменой не более 0,1% hard labels лучшего checkpoint. Предохранитель — 50 000 эпох. Статус `training_plateau` отличается от `fixed_epoch_budget`.

Первые 1 000 эпох повторяются с тем же seed, поскольку старые NPZ не содержали состояния Adam. В `prefix_comparison.json` записано фактическое совпадение меток и численная разница loss. Сжатые `trace.json.gz` сохраняют каждую эпоху без потерь; метрики и info ссылаются на одну трассу, а не дублируют её. Команды итоговой сборки HTML — в разделе 2.

## 11. Проверка исправленного технического ядра, 3 октября

После подготовки той же панели:

```sh
python -m unittest discover -s tests -v
python -m compileall -q sbercluster scripts
python scripts/verify_joint_refinement.py --output artifacts/joint-refinement-repeated --bootstrap 500
```

Последняя команда выполняет восемь ограниченных подгонок совместной модели: дорожные K2/K3/K4 и региональный K4, каждый из первоначального старта и сохранённого конечного разбиения. Параметры `α=0,25`, seed=1729 и максимум 50 проходов закреплены; перенос учитывает точное изменение обоих центров. Проверяются все допустимые одиночные переносы после остановки. Выходной каталог должен быть новым. DMoN и временная сетка этой командой не переобучаются.

Выход содержит метки, признаки, значения цели, внешнее сравнение с 500 парными региональными bootstrap-выборками, версии и копии исходных модулей, входные хеши и manifest. Проверенный снимок — [joint-refinement](../reports/technical-core-2026-10-03/joint-refinement/manifest.json); [изменения и границы проверки](../reports/technical-core-2026-10-03/README.md).

Текущие загрузчики сверяют идентичность панели и её manifest до вычислений. Сравнение KMeans требует завершённого запуска и уникальных ID. Ошибки и прерывания новых исследовательских запусков записываются в `status.json`; повторное использование требует согласованных настроек, источников и полного набора свидетельств. Экспорт сначала собирается во временном каталоге и появляется под окончательным именем только после успешной проверки. Исторические источники сохранены побайтно, поэтому новое ядро и старые архивы различаются по хешам; это не основание перезаписывать прежние результаты.

## 12. Применение замороженной модели без обучения

`FrozenProfileModel` применяет сохранённые центры и полную настройку признаков. Параметры не подгоняются на новых наблюдениях. Вход — CSV с `entity_id`, `territory_id`, `period` и шестью категориями исходной панели. Идентификатор должен иметь вид `tid_<territory_id>`, период — первое число месяца в формате `YYYY-MM-01`; повторные пары территория–месяц отклоняются.

```sh
python scripts/predict_frozen.py --model reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json --input data/processed/panel.csv --output artifacts/frozen-monthly
python scripts/predict_frozen.py --model reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json --input data/processed/panel.csv --aggregation annual --output artifacts/frozen-annual
python scripts/verify_frozen_inference.py --output artifacts/frozen-verification
```

Каждый выходной каталог должен быть новым. Годовое применение требует все 12 месяцев каждого года для каждой территории и использует медиану преобразованных месячных признаков. Проверочная команда сравнивает все месячные и оба годовых разбиения с архивом; обучение не запускается.

Для встроенного архивного пути модель автоматически проверяется по `SHA256.json`. Для собственной модели можно передать `--expected-model-sha256 <hash>`: ожидаемый хеш должен быть взят из доверенного источника. Без него произвольный модельный файл проходит структурную проверку, а его фактический хеш записывается в результат. Хеш-манифест связывает версии файлов; криптографической подписью автора он не является.

Результат содержит `assignments.csv`, точную копию модели, исходники использованных модулей, версии зависимостей и манифест хешей. `relative_distance_margin` — относительный разрыв расстояний до двух ближайших центров, а не вероятность. Новые территории разрешены и отмечаются `in_reference_cohort=false`; сопоставимость их границ и исходной статистики проверяется отдельно.

## 13. Сбой подготовки и восстановление

`python scripts/prepare.py` сначала проверяет источники и строит результаты во временном каталоге. Исходные файлы проверяются по хешам до чтения и перед публикацией. Одновременно разрешена одна подготовка на рабочую копию: файл `.prepare.lock` предотвращает второго писателя. При обычном сбое прежние результаты восстанавливаются, манифест заменяется последним. Отсутствующие координаты в официальном справочнике допустимы: они не входят в признаки модели.

При ошибке самого восстановления сохраняются каталог `.prepare-*` с резервными файлами и блокировка; исключение сообщает путь. Перед ручным восстановлением нужно убедиться, что процесс подготовки завершён, сверить резервные файлы и восстановить согласованный комплект данных с манифестом. Блокировку снимают после восстановления. Автоматически удалять её по возрасту нельзя. Набор заменяемых файлов не является единой транзакцией при отключении питания; потребители должны запускаться после успешного завершения подготовки.

## 14. Национальная внешняя проверка

После подготовки исходной панели в среде `requirements.txt`:

```sh
python -m scripts.download_external_national_archive
python -m scripts.prepare_external_national
python -m scripts.validate_external_national --permutations 999 --bootstrap 1999
python -m scripts.validate_external_national_robustness
python -m scripts.summarize_external_national
```

Загрузчик проверяет закреплённые SHA-256 архивов Росстата / «Если быть точным». Ключи связываются по наблюдаемому ОКТМО и году, без подстановки современного кода вместо исторического. Экономические исходы не участвуют в обучении кластеров. В обоих годах исходов используются фиксированные расходные профили 2023 года; экономические регрессии обучаются отдельно. Перестановочные p-values описывают только исследовательский случайный контроль; основные результаты — ошибки при исключении регионов и парные интервалы. [Полные определения, покрытие, первоисточник и ограничения](../reports/external-national-2026-10-03/README.md).

## 15. Применение годовой альтернативы Huber75

Основная модель остаётся прежней. Для отдельной годовой альтернативы с улучшенным силуэтом используется собственный формат и полное правило назначения, включающее смещения границ:

```sh
python -m scripts.predict_frontier --model reports/model-frontier-2026-10-03/models/sw_constrained_huber75.json --expected-sha256 41a1071c7d35019f93218d09b6b5abc293bec3b3b4fafdc740cf2c7ac572003a --panel data/processed/panel.csv --year 2023 --output runs/frontier-prediction
python -m scripts.verify_frontier_geometry --output runs/frontier-geometry-audit.json
```

Каталог предсказаний и файл аудита должны быть новыми. В выбранном году каждой территории нужны все 12 месяцев. Хешируется содержимое, которое действительно прочитано для применения; неверный формат или хеш отклоняется. Старый `predict_frozen.py` не принимает формат `prototype_frontier`.

Применение к новым годовым данным не гарантирует прежние размеры групп и индексы качества. Проверка отдельных месяцев обнаружила сезонные малые группы, поэтому альтернативе не приписывается гарантия месячной устойчивости. Выход — описание расходов, а не производственный тип или прогноз зарплаты. [Полный конечный эксперимент, конфигурации, контрастная альтернатива и команды обучения](../reports/model-frontier-2026-10-03/README.md).


## 16. Два временных канала и конечная проверка

Новый парный канал вычитает медиану изменений фиксированного референса; исходный канал сохраняется. Получение текущей поправки использует текущие записи референса, а применение зафиксированного пакета детерминировано. Оба режима описательные:2024 просмотрен, неизменность границ не доказана. [Создание референса, пакетов и применение с проверкой хешей](TEMPORAL_CORE.md).

```powershell
python -m scripts.benchmark_dual_profiles --output runs/dual-reproduced
python -m scripts.benchmark_temporal_profiles --output runs/temporal-reproduced
python -m unittest discover -s tests -p test_dual_profiles.py -v
python -m unittest discover -s tests -p test_temporal_profiles.py -v
```

Каталоги должны быть новыми. Второй benchmark повторяет ровно конечный набор трёх исследовательских вариантов; все три не прошли совместный отбор, новых рекомендуемых моделей он не создаёт. [Исходные свидетельства и обе геометрии](../reports/temporal-core-2026-10-03/README.md).

## 17. Полная региональная экономическая проверка и границы пропусков

Нужны исходная панель и национальные `cohort-2023.csv`, `cohort-2024.csv`, `frozen-labels.json` с хешами из архивного протокола. Зарплатный сценарий исключает регион из всех новых обучающих стадий. Отраслевой сценарий повторно использует расходные разбиения и предсказывает также неизвестные исходы полного состава с доступными контролями.

```powershell
python -m scripts.validate_economic_generalization --output runs/economic-generalization-reproduced --include-h75
python -m scripts.validate_economic_sector_generalization --expense-folds runs/economic-generalization-reproduced --output runs/sector-generalization-reproduced
python -m unittest discover -s tests -p "test_economic*.py" -v
```

Для проверки стоимости первого зарплатного разбиения добавьте `--max-new-folds 1`; дальнейшая команда без ограничения переиспользует только кэш совпадающего протокола/входов/исходников. Изменение снимка требует нового каталога. Все прогнозы 2024 в этом сценарии используют только 2023 признаки и 2023 обучающие зарплаты; прежняя национальная проверка заново обучала экономическую модель на исходах каждого года и сохранена как другой эксперимент.

Один показатель расходов в контролях — log медианы месячного TOTAL 2023, не годовая сумма. Условные зарплатные bootstrap-интервалы и точные отраслевые границы неизвестных долей имеют разные смыслы. [Метод и ограничения](ECONOMIC_GENERALIZATION.md) · [полный пакет и manifest](../reports/economic-generalization-2026-10-03/README.md).

## 18. Автономная проверка поставляемого ядра

```sh
python -m scripts.verify_technical_core --output artifacts/technical-acceptance.json
```

Нужен новый выходной файл. Команда проверяет четыре пакета свидетельств, сохранённые правила и реальные CLI на небольшом синтетическом вводе, без скачивания данных и обучения. Для одной целостности файлов доступен `--integrity-only`; предсказания в этом режиме имеют статус `NOT_RUN`. Подробный контракт и коды завершения описаны в [TECHNICAL_ACCEPTANCE.md](TECHNICAL_ACCEPTANCE.md).


## 19. Сетевая типология и проверка аналогов

Полное объяснение основной модели и её практической проверки находится в [NETWORK_TYPOLOGY.md](NETWORK_TYPOLOGY.md). Нужны зависимости `requirements-models.txt`, подготовленные входы из `configs/v12.json` и сохранённые `reports/v1.2/summary.json`, `labels.csv`, `methods.csv`, `model.npz`. Сжатая панель `data/v12/panel.csv.gz` содержит тот же состав 2 016 МО; исходные значения должны совпадать с подготовленной панелью.

```sh
python -m scripts.validate_v12_findings --output reports/v1.2
```

Чтобы сверить поставляемые таблицы, добавьте `--check`: команда повторит расчёт во временном каталоге и сравнит результаты и хеши с указанным пакетом.

Команда без `--check` обновляет исправленные производные таблицы в существующем пакете: `analogues.csv` и `analogue_comparisons.csv` на общей выборке 2 016 МО, отдельные `*_diagnostic.csv` на 2 011 МО, прогнозы каждого МО, динамику групп и сводный JSON. Ошибки роста считаются в процентных пунктах, логарифмический рост сохраняется для геометрического среднего динамики групп. Строгое исключение 73 регионов заново обучает фиксированный вариант по 2023 и пишет `strict-region-validation`; `leave_one_region_out.csv` содержит строгие метрики, прежняя проверка только экономической регрессии сохранена внутри отдельного пакета. Проверка отсутствия влияния исключённого региона сравнивает хеши обучающего состояния до и после изменения его входов. В этой команде 2024 не обучает региональные модели, исходы 2025 не читаются.

`validate_v12_findings --output runs/network-findings-reproduced` создаёт отдельную копию сохранённого пакета и повторяет исправленные проверки. Эта команда не пересчитывает историческую сетку отбора. Для полного повторения сравнения признаков, девяти правил рёбер, K и десяти методов используйте новый каталог:

```sh
python -m scripts.run_v12 --config configs/v12.json --output runs/network-typology-reproduced
```

Флаг `--quick` у `scripts.run_v12` уменьшает число повторений и служит пробным запуском; его результаты не совпадают с полной серией. Параметры полного сравнения закреплены в конфиге, входные хеши и версии записаны в `provenance.json`. Выбор рёбер использует внешние показатели и совпадение сетей 2023–2024; выбор метода также использует оба года. Строгие региональные разбиения повторяют уже выбранный алгоритм, вложенный отбор архитектуры не проводится. [Расчётный пакет](../reports/v1.2/summary.json).

Повторная проверка 8 октября 2026 года на Windows воспроизвела побитово все 73 хеша обучающих состояний и назначения всех 2 016 исключённых МО. [Протокол с версиями, входными хешами и результатом каждого региона](../reports/v1.2-added-value/windows-state-verification.json) относится к фиксированной модели; полная сетка отбора и внешние экономические исходы в этом запуске не пересчитывались.

Координаты шести признаков для браузера округляются до 10 знаков после запятой, чтобы Windows и Linux собирали одинаковый файл данных. Список соседей и научные метрики вычисляются до этого округления.

## 20. Переотрисовка атласа из сохранённых данных

После правки интерфейса или поясняющих текстов можно собрать атлас из уже поставляемых данных:

```sh
python -m scripts.render_saved_atlas
```

Команда читает `docs/index.html` и его файл данных из `docs/assets`, проверяет хеш, обновляет текстовые разделы по сохранённым таблицам `reports/v1.2` и применяет шаблон. Метки, признаки, соседи и геометрия берутся из прежнего файла данных. Подготовка исходной панели и повторное обучение для этой команды не нужны.

Перед обновлением текста проверяются шесть хешей отчётов, закреплённых в атласе. Если файл изменён или хеш отсутствует, команда останавливается до записи результата. Эти хеши не охватывают все вспомогательные таблицы; команда предназначена для текущего согласованного набора файлов проекта.

Для отдельного автономного HTML укажите новый путь:

```sh
python -m scripts.render_saved_atlas --output preview/atlas.html --standalone
```

В этом варианте данные и шрифты включены в HTML. Прежние файлы данных сохраняются. Для повторения самих научных расчётов используйте предыдущие разделы инструкции.
