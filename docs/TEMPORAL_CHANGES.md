# Проекция профиля и ретроспективные изменения среднего

Годовая графовая группа и ближайший центроид — разные правила. Теперь baseline
равен `q(reference)`, месячная метка равна `q(month)` в той же системе координат,
при тех же замороженных центрах. Графовая группа сохраняется отдельным полем.
При равных расстояниях выбирается первый центр. `projection_disagreement`
описывает различие методов; это не событие. `persistent_projection_change`
означает, что последние шесть месячных меток одинаковы и отличаются от
проекционного baseline. Это пересечение геометрической границы, без
подтверждения экономического перехода.

Исторические `reports/v1.2` и synthetic executed-20261009 сохранены. В частности,
48 ложных флагов на постоянных реальных профилях относятся к прежнему сравнению
с графовой меткой. Новое исправление не меняет старые результаты задним числом.

Отдельный исследовательский слой использует `ruptures==1.1.10`,
`Pelt(model='l2', jump=1, min_size=3)`. Он минимизирует сумму внутрисегментных
квадратов отклонений с положительным штрафом за сегментацию. `boundaries` —
нулевые индексы начала нового сегмента, без терминального T. Например, индекс12
соответствует январю2024 в панели2023–2024. `persistent_boundaries` требует длину
следующего сегмента не менее6. Последние три месяца дают censored candidate.
Это анализ всех24 месяцев сразу, без измеренной задержки онлайн-обнаружения.
Величина эффекта — разность средних соседних оценённых сегментов; не вероятность.

Перед подбором штрафа замораживаются SHA кода, конфигурации, входов, списка
seeds и purpose map в `registration.json`. Технический pilot использует только
fixture seed610090001, один c=1, оценивает формат/время и не выбирает параметр.
Для каждого из16 tune seeds обрабатываются12 клеток,2 канала и6 объявленных c.
Выбор читает только применимые null клетки. Самый малый c с FPR<=.05 в каждой
null клетке закрывается в `selected.json` и его SHA до первых64 evaluation seeds.
min_size6 и дополнительные stress сценарии отложены до fit по конфигурации.
Если c не выбран, evaluation этого канала сохраняет все клетки со статусом
`NO_CALIBRATED_CANDIDATE`; PASS не присваивается и grid не расширяется.

Синтетика:32 объекта,24 месяца,5 координат, Gaussian means/noise, AR0.5 с
стационарной первой точкой и std инноваций sqrt(1-rho²); N0 использует IID.
Общая годовая sin/cos компонента имеет амплитуду1 в первых двух координатах.
Её оценка — среднее по фиксированной cohort первых12 entity-centered месячных
наблюдений. Offset повторяется во втором году. При одном году оценка содержит
общий тренировочный шум. N1 и N2 имеют одинаковые AR данные и residuals: N2
выделяет null сезонного механизма, это не дополнительная независимая репликация.
Изменяемые8 объектов выбираются отдельным RNG purpose. Первые12 месяцев
в совпадающих AR сценариях тождественны; N0 и boundary means N5 объявлены отдельно.
Детектор не получает truth. Абсолютный канал сохраняет общий сдвиг, относительный
вычитает текущую медиану фиксированной полной cohort и сохраняет её в common_drift.
Относительный канал transductive, зависит от состава cohort и может скрывать
сдвиг большинства объектов. Таких broad stresses в primary нет.

Штраф `max(1e-12,c*d*variance_train*log24)` использует variance отдельно каждого
канала: сумма квадратов после entity means первых12 наблюдений, делённая на
N*d*11. Calendar, center/scale и variance не подгоняются по2024.
Real projection использует прежний pinned six-coordinate v12 transform с
помесячной contemporaneous медианой уровня/общего профиля: это описательное
исправление его baseline, без обещания train-only прогноза. Реальный PELT
использует отдельно five-coordinate log(category/TOTAL), median/IQR/sqrt5,
calendar/variance обучены только по2023. Уровень расходов не входит в PELT.
Категории — относительные интенсивности; замыкание бюджета/CLR/ILR не применяются.
В legacy `model.npz` поле level_center отсутствует: этот исторический reference
детерминированно восстанавливается как помесячная медиана log(total) той же
замороженной полной cohort. Эквивалентность всех48384 месячных меток старому
labels.csv проверяется отдельным тестом и в real exporter. Это совместимость
старой описательной проекции, без изменений генератора/штрафа/новой CP calibration.

Matching — максимальное число пар boundary/truth с допуском±1 индекс, без
повторного использования точки; при равенстве минимизируется сумма ошибок.
Recall при отсутствии truth равен null. Precision без предсказаний равен null.
Для P6 вход12 и возврат18 сохраняются раздельными счётчиками; persistent recall
считает обе eligible границы, а не только вход. C3 boundary21 включён в all-CP
score и excluded из persistent recall denominator. Spike null относится именно
к persistent событиям, transient границы сохраняются. False boundaries/object-year
считаются для всех CP. Means of per-seed metrics используют percentile paired
bootstrap10000 с одинаковыми seed-resampling indices по клеткам; undefined
precision пропускается, число truth/predicted сохранено. По одному заранее
выбранному representative object/seed/cell строится one-sided95% Clopper–Pearson
FPR upper. Это per-cell inference, без simultaneous guarantee, real-world FWER
и независимости муниципалитетов. Вырожденный bootstrap0..0 не доказывает нулевой риск.

Для synthetic promotion нужны инварианты, selected c и в каждой null клетке
mean FPR<=.05 плюс representative upper<=.05. Power публикуется при любом исходе,
универсального recall80% нет. Даже PASS ограничен данным генератором. На реальных
данных2024 повторно использован, истинных event labels и независимой monthly
проверки территориальных границ нет. Реальные CP всегда обозначены
`RETROSPECTIVE_RESEARCH_CANDIDATE`, без p-values/recall/confirmed transitions.
Канонические ID,24 наблюдения и исходные2016 геометрии сохранены.

Запуск из корня:

```powershell
python -m pip install -r requirements-temporal.txt
python -B -X utf8 -m unittest discover -s tests -p test_temporal_changes.py -v
python -B -X utf8 -m scripts.benchmark_temporal_changes --pilot-only --pilot-max-seconds 120 --output artifacts/temporal-v4-pilot
python -B -X utf8 -m scripts.benchmark_temporal_changes --pilot-report artifacts/temporal-v4-pilot/pilot.json --output reports/temporal-v4
python -B -X utf8 -m scripts.verify_temporal_changes
```

Benchmark требует новый output и отказывается перезаписывать результаты.
Standalone pilot не читает реальные scientific inputs и не начинает tune/eval.
Его120s timer проверяется между клетками, не прерывает Python imports; elapsed
включает deterministic fixtures. Основной запуск может повторно использовать
pilot только при совпадении всех code/config SHA, без лишнего пересчёта.
Лимит60 минут проверяется между seeds, checkpoint сохраняет все исходы;
timeout имеет статус INCOMPLETE. Полные real CP и synthetic events лежат в
`real-correction.json`/`events.csv`, numeric table в `real-monthly-projections.csv`.
Контракт интеграции: `revision`, `periods`, `entities` keyed canonical entity_id,
`common_drift` и `summary`. В entities находятся сохранённая graph_label,
projected_baseline_label, projected_labels[24], distance_margin[24],
projection_disagreement, boundary_crossed[24], persistent_projection_change,
observation_status/mask и cp.absolute/cp.relative с границами/эффектами/штрафом.
Verifier проверяет полноту/неизменность регистрации и численную согласованность;
exit0 означает корректные сохранённые результаты, а не пригодность мониторинга.
