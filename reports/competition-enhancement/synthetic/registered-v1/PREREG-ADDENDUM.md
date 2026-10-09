# Дополнение перед запуском science-plan-v1

Зафиксировано до собственных новых исходов на SberAI и до tune/evaluation синтетики. Исходные PLAN.md и design.json сохранены без изменения:
PLAN SHA256=73d98737ef8dea1a5e40a77ab9a501a84c56005ced6e68827838e2399121d217.
design SHA256=39132d0f10f00b74e4fc90c93343c5a2a503defcf8a09a8bba519cc730e6db27.

Изменение порядка выполнения утверждено продолжением поручения лида: ранняя копия C сохраняется отдельно; формальная приёмка C не блокирует разрешённые точные проверки и последующие научные запуски после выделения вычислительного слота. Тяжёлый слот назначает только лид. Процесс запускается один раз, в один поток и BelowNormal, с логом CPU, каждого физического диска, GPU и RAM; при95% собственная работа ждёт снижения нагрузки. Это не меняет условия научного успеха или pinned core.

## Основание дополнений

Лид передал содержательные идеи внешней Pro-консультации до новых исходов нашей работы. Это экспертные предложения, а не наши измерения. К ним добавлена независимая проверка объёма science_plan_review. Неизменённый видимый innerText сохранён лидом в C:/Users/Andreyka/Documents/Codex/2026-10-08/goal-goal-c-users-andreyka-documents-2/work/pro-methodology/consultation/RETURN.rendered.txt. Исходные RETURN.md и ZIP недоступны; состояние передачи CLOSED, локальные real2016 измерения ещё NOT_RUN. Эта запись не заменяет внешний ответ и не утверждает, что его предложения были нашими измерениями.

Проверка main alpha0 сохраняет KMeans baseline без дополнительных SSE-only переносов. Поэтому сравнение joint с alpha0 или KMeans не отделяет графовый штраф от изменения оптимизатора. Добавляем feature_hartigan: точная оптимизация только SSE из того же лучшего KMeans(n_init50) старта; тот же seed порядка объектов, запрет опустошения, max_sweeps200 и strict normalized decrease threshold1e-14. Это отдельная исследовательская функция runner, pinned joint не изменяется. В отчёт входят both terms J и sizes.

Добавляем два matched topology контроля только на evaluation seeds: weak_independent_permuted и weak_derived_permuted. Нормированный X, latent truth и исходная W совпадают с соответствующей основной клеткой. Вершины W переставляются независимым SeedSequence([root,generator,6,original_graph_family_id]); распределение степеней, плотность и неименованная топология W сохраняются. Перестановка не подбирается по исходу. Передаются оба графа, результаты на одинаковом X. Эти клетки не участвуют в подборе весов: глобальный tune остаётся на исходных12 клетках.

Оба контроля обязательны для содержательного вывода о роли сети. Новые сравнения являются заранее объявленными механистическими описательными сравнениями с парными bootstrap интервалами. Исходные четыре C1–C4 и их Holm поправка сохраняются. Не выдавать отдельный удобный интервал новых клеток за исходный подтверждающий тест.

До большой матрицы запускается diagnostics: actual2016 X_base, закреплённые annual labels и centers из model.npz; centroids пересчитываются для проверки совпадения с pinned центрами. Считать r0=mean(nearest(X_base,C)!=annual_labels). Затем повторить X_base36 раз без какого-либо изменения объектов и вызвать реальные assign_monthly и persistence(window6); проверить raw и national-adjusted варианты. Annual labels здесь reference assignment, не независимая экономическая истина. Число persistent disagreements на таком вводе показывает несогласованность этапов; оно не является оценкой ложных реальных экономических переходов.

Малые exact fixtures разрешены отдельно до science run. Уже выполнены generator invariants и ручные scorer controls, без сравнительного качества на tune/evaluation seed. Контроль Pro с X=(0,1,1.1,2.1), annual labels=(0,1,0,1) воспроизвёл 50% nearest-centroid disagreement и2 flags на неизменных рядах. Это ручной механизм, не новое наблюдение на МО. Для perfect external labels при onset19 scorer даёт recall1 и confirmation delay5; frozen labels дают recall0 и post-change destination accuracy0. Эти проверки не изменили критерии после исходов.

## Уточнение оценки нагрузки

В исходном верхнем объёме была пропущена месячная free re-clustering диагностика и стресс-примеры. Базовый год для них НЕ fit повторно. Free raw KMeans labels рассчитываются на adjusted X один раз на evaluation root seed, strength и regime, затем отдельные дешёвые Hungarian выравнивания относительно annual labels каждого метода. Tune траектории не рассчитываются: подбор основан только на static ARI.

Для primary32 seeds,2 strength,4 regimes,36 месяцев,n_init20:
32*2*4*36=9216 complete monthly fits,184320 стартов.
Stress только weak,2 режима,32 seeds:
32*2*36=2304 complete monthly fits,46080 стартов.
Итого11520 complete fits,230400 дополнительных стартов. Graph permutations и дополнительные методы используют тот же cached raw monthly fit и не увеличивают эту часть.

Переставленные графы добавляют128 evaluation datasets; их fixed и tuned row methods дают верхние256*50=12800 KEFRiNe стартов плюс128*50=6400 current-like стартов. При учёте KMeans per-cell верхняя оценка ещё6400 стартов; на деле baseline caches общие по X и graph controls не вызывают их заново. Консервативная верхняя оценка основного блока становится131200 стартов вместо105600. С free diagnostics полный консервативный предел361600 стартов, плюс graph-only320 fits и tiny checks. Feature-Hartigan кешируется вместе с KMeans по X и не имеет новых random initializations. Верхняя оценка не является числом реально исполненных вызовов; время измеряется на фактическом процессе.

Точный бюджет реального runner с его кешем можно закрепить до подбора. Пусть a=1, если selected joint alpha=.5, иначе a=2; b=1, если selected KEFRiNe beta=1, иначе b=2. Это только число уникальных параметров на evaluation dataset, а не новая свобода выбора. Tune datasets=8*2*6=96, evaluation datasets=32*2*8=512. Кеш X имеет2 strength*2 generators на каждый root seed, поэтому базовый KMeans вызывается32 раза на tune и128 на evaluation, всего160 fits и8000 стартов. Current-like вызывается96+512=608 раз,30400 стартов. Faithful solver вызывается96*4+512*b=384+512*b раз,44800 или70400 стартов. Free monthly KMeans имеет ровно11520 fits и230400 стартов. Итого randomized starts=313600 при b=1 или339200 при b=2. Консервативная прежняя оценка361600 остаётся корректной верхней границей, но реальный кеш уменьшает её.

Дополнительно без random starts: feature-Hartigan160 вызовов; Potts single-vertex optimization384+512*a вызовов; graph-only spectral320 вызовов; alpha0 production reuse608 вызовов без дальнейшей оптимизации. Всего optimizer calls включая spectral и SSE-only Hartigan14560+512*(a-1)+512*(b-1), то есть14560–15584;608 alpha0 no-op учитываются отдельно. Alias fixed/tuned при равных параметрах дают две строки метрик с одним fit. Эти числа проверяются статически по runner; после реального опыта selected.json и per-fit ledger должны подтвердить конкретную ветвь. Время до запуска не измерено. Максимальные итерации и sweep caps остаются300 и200.

Primary tracked models теперь8 (прежние7 плюсfeature_hartigan); core evaluation графов8. Поэтому adjusted primary trajectories=32*8*4*8=8192. Stress на2 исходных графах=32*2*2*8=1024. Raw projection диагностика повторяет assignments без новых fits. Free alignment считается отдельно. Frozen KMeans negative control имеет только assignments и scoring. Все эти результаты подписаны variant; main adjusted tracking не подменяется free или raw. Полное число строк temporal metrics с adjusted,raw,free и freeze равно32*(8*4+2*2)*(8*3+2)=29952. Истинные события и строки сообщений сами по себе новых fits не вызывают.

11520 месячных fits НЕ разрешены до завершения ранних diagnostics, проверки основных --check и выделения нового science слота. Runner предлагает diagnostics и один run после них; run проверяет SHA diagnostics той же регистрации. По поручению лида можно отдельно зарегистрировать более короткое выполнение до просмотра evaluation исходов; текущий runner ничего не урезает и не открывает данные2025.

Уточнение управления ресурсами дано лидом до регистрации и собственных исходов: C или другой физический диск95% не требует вечного ожидания всех фоновых приложений. Наш процесс снижает CPU priority до Idle, I/O priority до VeryLow где psutil поддерживает, делает ограниченную паузу1секунду перед своей работой и пишет это в монитор. Основной full run при CPU,RAM илиGPU95% приостанавливает compute. Малые exact checks и diagnostics без optimizer fits при фоновомCPU илиGPU pressure выполняются в Idle, а приRAM95% ждут. Это адаптация нагрузки; данные, бюджеты и критерии качества сохраняются.

## Команды и версии

До регистрации и просмотра научных исходов независимый рецензент обнаружил пропуск обещанных PLAN.md метрик macro accuracy и accuracy неизменившихся объектов в выходе runner. Он исправлен без смены модели или критерия: macro_state_accuracy усредняет доли верных состояний отдельно по истинным классам каждого месяца13–36, затем по месяцам. Классы без объектов в соответствующем месяце не входят в его среднее; в зарегистрированных генераторах все четыре класса присутствуют. unchanged_state_accuracy считает долю верных состояний за те же месяцы только у объектов, чья внешняя истина не менялась. Обе метрики используют единственную базовую Hungarian mapping. Уже существующая changed_destination_accuracy сохраняется и добавляется в сводку вместе с ними. Ручной broad fixture с постоянным назначением в0 должен дать macro_state_accuracy=.25, state_accuracy=.475 за месяцы13–36 и unchanged_state_accuracy=.25; конечная micro accuracy=.55 не заменяет macro.

Рабочий каталог:
C:/Users/Andreyka/Documents/Codex/2026-10-08/goal-goal-c-users-andreyka-documents-2/work/repo.
Интерпретатор:
C:/Users/Andreyka/AppData/Local/Programs/Python/Python310/python.exe.

Сначала register фиксирует точные SHA runner, faithful module, imported scientific core, исходных планов и этого дополнения; сверяет132 oracle pins и package versions по reports/environment.json. Каталог регистрации должен быть новым. После записи registration.json исходники не меняются до диагностики и полного опыта; существенная ошибка требует новой регистрации и явного сохранения старого исхода.

В PowerShell для подготовки:
$scienceW='C:/Users/Andreyka/Documents/Codex/2026-09-23/gh/work/SberAI/.swarm/atlas-competition-20261008-v2'
$sciencePython='C:/Users/Andreyka/AppData/Local/Programs/Python/Python310/python.exe'
& $sciencePython -X utf8 -m scripts.competition_synthetic register --design "$scienceW/agents/science_design/design.json" --plan "$scienceW/agents/science_design/PLAN.md" --pins "$scienceW/oracle/pins.json" --addendum "$scienceW/agents/science_design/PREREG-ADDENDUM.md" --output reports/competition-enhancement/synthetic/registered-v1

Первый science слот после сохранения C:
& $sciencePython -X utf8 -m scripts.competition_synthetic diagnostics --registration reports/competition-enhancement/synthetic/registered-v1/registration.json --output reports/competition-enhancement/synthetic/registered-v1/diagnostics --slot-id ACTUAL_LEAD_SLOT_ID

Единственный полный зарегистрированный опыт после разрешения лида:
& $sciencePython -X utf8 -m scripts.competition_synthetic run --registration reports/competition-enhancement/synthetic/registered-v1/registration.json --diagnostics reports/competition-enhancement/synthetic/registered-v1/diagnostics/diagnostics.json --run-dir reports/competition-enhancement/synthetic/registered-v1/run --slot-id ACTUAL_LEAD_SLOT_ID

Реальный slot_id задаёт лид, исполнитель его не выдумывает. register не означает научного выполнения. Runner создаёт новый output и отказывается от существующего; ошибочный лог не перезаписывается. Перезапуск после исправления требует нового каталога и объяснения. Один full run последовательно делает tune, сохраняет selected.json с SHA tune metrics, затем evaluation и report; evaluation не может незаметно поменять selected.json.

Результаты: model-input*.npz и evaluator-truth*.npz, per-seed fit JSON с ledger faithful starts и convergence, checkpoint, static-metrics.csv, temporal-metrics.csv, events.csv, graph-diagnostics.csv, compressed labels, selected.json, contrasts.csv, confidence-intervals.csv, mechanistic-comparisons.csv, summary.csv, environment.json и resource-monitor.jsonl. Все seed сохраняются. Ошибка сохраняется в failure.json и прерывает процесс; результат не называется COMPLETED. Main core, критерии и официальные scientific pins остаются исходными.

