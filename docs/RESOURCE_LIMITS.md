# Запуск с ограничением нагрузки

Для небольших экспериментов используется `scripts/run_bounded.py`. Он запускает один процесс расчёта и сохраняет журнал нагрузки в `artifacts/resource_runs/`.

- CPU: Windows Job Object ограничивает процесс и его дочерние процессы до 20% процессорного времени. Дополнительно выбраны два логических процессора и два потока вычислительных библиотек.
- Память: суммарная частная выделенная память процессов задачи ограничена 1 ГиБ. Это отдельная величина от всей занятой оперативной памяти компьютера.
- GPU: выбранные методы выполняются на CPU. Видеокарта для этого запуска не используется.
- Работа чередуется с короткими паузами: 0,15 секунды вычислений и 0,35 секунды паузы. Это помогает распределить чтение файлов на HDD; измеренный предел активности диска остаётся обязательным.
- Раз в секунду измеряется общая загрузка CPU, RAM, GPU, видеопамяти и каждого физического диска. При значении выше 70% расчёт останавливается. Высокая нагрузка до старта блокирует запуск.

Мониторинг диска не является жёстким ограничением: короткий всплеск между замерами возможен. Другие приложения также влияют на общую загрузку и могут вызвать остановку. Ограничения относятся к вычислительной задаче и не меняют настройки остальных приложений.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-monitor.txt
.\.venv\Scripts\python.exe scripts/run_bounded.py --config configs/pilot_bounded.json
```

Конфигурация `pilot_bounded.json` включает только декабрь 2023 года и три статических метода. Временная кластеризация в этом запуске не используется.

Механизмы Windows: [ограничение CPU](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information), [ограничение памяти](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information), [показатели нагрузки дисков](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw). `SetIoRateControlInformationJobObject` [не поддерживается в современных Windows](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-setioratecontrolinformationjobobject), поэтому он не используется для обещания жёсткого лимита диска.
