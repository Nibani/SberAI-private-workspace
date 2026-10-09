"""Small evidence panels rendered from current saved results, with explicit scope."""
from __future__ import annotations

import json
from pathlib import Path

PUBLIC = 'https://github.com/Nibani/sber-public/blob/main/'


def render_economics(path: Path) -> str:
    atlas = json.loads(path.read_text(encoding='utf-8'))
    summary = json.loads(path.with_name('summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'COMPLETE':
        raise ValueError('The current economic experiment is incomplete')
    covered = sum(row.get('type') is not None for row in atlas['entities'].values())
    total = len(atlas['entities'])
    delta = summary['type_contrast']
    gain = 100 * float(delta['estimate'])
    low = 100 * float(delta['lower97_5'])
    result = ('Типы прошли заранее заданную проверку добавочной точности.'
              if summary['type_external_increment'] else
              'Добавочная точность типов сверх сильного непрерывного описания не подтверждена заданной проверкой.')
    number = lambda value: f'{value:.2f}'.replace('.', ',')
    compact = summary['compressed_contrasts']['C4_vs_C']
    graph = json.loads(path.with_name('graph-controls.json').read_text(encoding='utf-8'))
    profiles = []
    for item in atlas['types']:
        p = item['profile']
        profiles.append(f'<tr><th scope="row">Профиль {int(item["id"]) + 1}</th>'
                        f'<td>{int(item["n"])}</td>'
                        f'<td>{number(100 * p["organization_jobs_per_resident"])}</td>'
                        f'<td>{number(100 * p["OPQ_headcount_share"])}%</td></tr>')
    rows = []
    labels = {'PERSIST': 'Зарплата предыдущего года', 'GROWTH': 'Предыдущая зарплата + общий рост',
              'PQUAD': 'Непрерывные признаки + предыдущая зарплата',
              'PQUAD4': 'Те же признаки + четыре типа',
              'PQUADG': 'Те же признаки + графовые прототипы'}
    for method, label in labels.items():
        value = summary['losses'][method]['region_mse']
        exact = f'{value:.6f}'.replace('.', ',')
        rows.append(f'<tr><th scope="row">{label}</th><td>{exact}</td></tr>')
    return (
        '<section class="card v12-card" id="economic-structure" aria-labelledby="economic-structure-title">'
        '<div class="section-head"><div><h2 id="economic-structure-title">Четыре профиля организаций и потребления</h2>'
        '<p>Сопоставляем расходные особенности с плотностью работников организаций и долей '
        'госуправления, образования и здравоохранения. Зарплата не участвует в построении типов.</p></div></div>'
        f'<p>Трудовые признаки доступны для {covered} из {total} территорий. При пропусках '
        'экономический тип не присваивается; расходный профиль остаётся доступен.</p>'
        f'<p>С компактной моделью уровня расходов, населения и статуса МО четыре типа '
        f'снижают ошибку оценки зарплаты на <strong>{number(100 * compact["estimate"])}%</strong>. '
        f'Нижняя односторонняя граница 97,5%: {number(100 * compact["lower97_5"])}%.</p>'
        f'<p><strong>{result}</strong> Относительное уменьшение средней квадратичной ошибки: '
        f'{number(gain)}%; нижняя односторонняя граница 97,5%: {number(low)}% '
        '(положительное значение означает уменьшение ошибки). Порог до расчёта: 5%.</p>'
        '<details class="comparison-details"><summary>Сравнение на внешнем показателе</summary>'
        '<div class="details-body"><p>Четыре типа описывают разные сочетания работников организаций '
        'и расходных особенностей. Показатели ниже — медианы территорий каждого типа.</p>'
        '<div class="table-scroll" tabindex="0"><table class="v12-table">'
        '<thead><tr><th scope="col">Тип</th><th scope="col">МО</th>'
        '<th scope="col">Работников на 100 жителей</th><th scope="col">Доля O/P/Q</th></tr></thead>'
        f'<tbody>{"".join(profiles)}</tbody></table></div>'
        '<div class="table-scroll" tabindex="0"><table class="v12-table">'
        '<thead><tr><th scope="col">Способ оценки зарплаты 2024 года</th>'
        '<th scope="col">MSE логарифма зарплаты</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<p>В каждой проверке целые регионы отложены. Признаки и типы обучаются по 2023 году; '
        'оценка зарплаты использует 2024 год только в остальных регионах. Это ретроспективная '
        'оценка для новых регионов, не прогноз из 2023 года. Каждый регион имеет одинаковый вес; '
        'интервалы условны на полученных моделях. Эти данные уже изучались раньше.</p>'
        f'<p>Отдельная временная сеть сохранила сходство месячных отклонений в следующем году: '
        f'{number(graph["graph_score"])} против {number(graph["static_score"])} у соседей '
        'по статическим признакам. Это дополнительная информация о совместной динамике той же панели. '
        'Графовые группы не прошли все условия допуска и не дали прибавки в оценке зарплаты.</p>'
        '<p>Отраслевая доля не определяет форму собственности организаций. Число работников '
        'относится к охваченным организациям и сопоставляется со всеми жителями; '
        'это не уровень занятости населения.</p>'
        f'<p><a href="{PUBLIC}docs/CONDITIONAL_PROFILES.md">Типы, охват и интерпретация</a> · '
        f'<a href="{PUBLIC}reports/conditional-v4/summary.json">Все сравнения и условия допуска</a> · '
        f'<a href="{PUBLIC}reports/conditional-v4/predictions.csv">Предсказания для каждой территории</a>'
        '</p></div></details></section>'
    )


def load_temporal(path: Path) -> tuple[dict, dict]:
    real = json.loads(path.read_text(encoding='utf-8'))
    summary = json.loads(path.with_name('summary.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'COMPLETE':
        raise ValueError('Current temporal evidence is incomplete')
    if real['summary']['n_entities'] != len(real['entities']):
        raise ValueError('Temporal cohort count differs from saved entities')
    if real['summary']['constant_fixture_corrected_projection_flags'] != 0:
        raise ValueError('Constant profiles must not produce projected changes')
    return real, summary


def render_temporal(path: Path) -> str:
    real, result = load_temporal(path)
    s = real['summary']
    n = s['n_entities']
    persistent = s['actual_persistent_projection_changes']
    mismatches = s['projection_disagreements']
    gate = result.get('synthetic_promotion_gate')
    gate_note = ('Заранее заданные ограничения ложных тревог на синтетике пройдены.'
                 if gate == 'PASS' else
                 'Надёжность детектора как системы мониторинга пока не подтверждена всеми заданными проверками.')
    return (
        '<section class="card v12-card" id="v12-dynamics" aria-labelledby="v12-dynamics-title">'
        '<div class="section-head"><div><h2 id="v12-dynamics-title">Изменился профиль или только метка?</h2>'
        '<p>Сравниваем исходный и месячный профили одним правилом — по ближайшему фиксированному центру. '
        'Годовая графовая метка хранится отдельно.</p></div><span class="tag">2023–2024</span></div>'
        '<div class="metrics v12-metrics">'
        f'<div class="metric"><b>{persistent}</b><small>из {n} территорий остаются у другого центра '
        'все последние шесть месяцев</small></div>'
        f'<div class="metric"><b>{mismatches}</b><small>годовых графовых меток отличаются '
        'от проекции того же исходного профиля</small></div>'
        '<div class="metric"><b>0</b><small>ложных смен на полностью неизменной панели '
        'при одинаковом правиле сравнения</small></div></div>'
        '<p>Переход через границу между центрами — повод посмотреть данные. Он ещё не доказывает '
        'перестройку местной экономики: нужно проверить охват платежей, границы территории и внешние показатели.</p>'
        '<details class="comparison-details"><summary>Как проверен поиск устойчивых изменений</summary>'
        f'<div class="details-body"><p>{gate_note} Проверки отделяют сезонность, общую волну, одиночный '
        'выброс и устойчивый сдвиг. Настройка штрафа и оценка качества используют разные случайные последовательности.</p>'
        '<p>Реальные точки изменения — исследовательские кандидаты. Истинные даты экономических событий '
        'для этой панели неизвестны; 2024 год уже изучался в предыдущих опытах.</p>'
        f'<p><a href="{PUBLIC}reports/temporal-v4/summary.json">Все результаты и ограничения детектора</a> · '
        f'<a href="{PUBLIC}docs/TEMPORAL_CHANGES.md">Методика</a> · '
        f'<a href="{PUBLIC}docs/evidence-index.md#legacy">Исторические расчёты</a></p></div></details></section>'
    )


def render_monitoring(path: Path) -> str:
    real, _ = load_temporal(path)
    count = real['summary']['actual_persistent_projection_changes']
    return ('<h3 class="v12-h3">Что проверить на территории</h3>'
            f'<p>У {count} территорий последние шесть месяцев относятся к другому ближайшему профилю, '
            'чем исходный профиль 2023 года. Это список для исследования, а не реестр подтверждённых '
            'экономических переходов. Начните с месячного ряда и сравнения с аналогами.</p>'
            f'<p><a href="{PUBLIC}reports/temporal-v4/real-monthly-projections.csv">'
            'Одинаковое правило назначения: все территории и месяцы</a></p>')
