# ML_OPS_2023



### Решаемая задача
Классификация ирисов через sklearn - RandomForestClassifier.

### Системная конфигурация
* Apple - Macbook Pro 13 Retina Touch
* 2 GHz 4‑ядерный процессор Intel Core i5
* 16 ГБ 3733 MHz LPDDR4X
* macOS Ventura 13.4

### Дерево model_repository
```
.
├── Dockerfile
├── assets
├── docker-compose.yaml
├── model_repository
│   ├── argmax-model
│   │   ├── 1
│   │   │   └── model.py
│   │   └── config.pbtxt
│   ├── ensemble-onnx
│   │   ├── 1
│   │   └── config.pbtxt
│   └── onnx-model
│       ├── 1
│       │   └── model.onnx
│       └── config.pbtxt
└── requirements.txt
```
### Метрики throughput и latency
Будет измеряться следующим образом:
perf_analyzer -m onnx-model -u localhost:8500 --concurrency-range 1:16 --shape X:1,4
* До оптимизаций (лучшая версия из не стандартных):
```
Concurrency: 1, throughput: 2027.46 infer/sec, latency 492 usec
Concurrency: 2, throughput: 3945.43 infer/sec, latency 506 usec
Concurrency: 3, throughput: 5626.86 infer/sec, latency 532 usec
Concurrency: 4, throughput: 6796.04 infer/sec, latency 587 usec
Concurrency: 5, throughput: 8396.28 infer/sec, latency 594 usec
Concurrency: 6, throughput: 9631.18 infer/sec, latency 622 usec
Concurrency: 7, throughput: 10180.8 infer/sec, latency 686 usec
Concurrency: 8, throughput: 11182.9 infer/sec, latency 714 usec
Concurrency: 9, throughput: 11721.8 infer/sec, latency 766 usec
Concurrency: 10, throughput: 13502.6 infer/sec, latency 739 usec
Concurrency: 11, throughput: 14077.9 infer/sec, latency 780 usec
Concurrency: 12, throughput: 14483.1 infer/sec, latency 827 usec
Concurrency: 13, throughput: 16100.2 infer/sec, latency 806 usec
Concurrency: 14, throughput: 15977.6 infer/sec, latency 875 usec
Concurrency: 15, throughput: 17347.3 infer/sec, latency 863 usec
Concurrency: 16, throughput: 16540.2 infer/sec, latency 966 usec
```
* После оптимизаций(самая обычная):
```
Concurrency: 1, throughput: 2654.08 infer/sec, latency 376 usec
Concurrency: 2, throughput: 5242.76 infer/sec, latency 380 usec
Concurrency: 3, throughput: 8224.02 infer/sec, latency 364 usec
Concurrency: 4, throughput: 9556.46 infer/sec, latency 417 usec
Concurrency: 5, throughput: 9708.2 infer/sec, latency 514 usec
Concurrency: 6, throughput: 11608.9 infer/sec, latency 515 usec
Concurrency: 7, throughput: 12572.5 infer/sec, latency 555 usec
Concurrency: 8, throughput: 13452.1 infer/sec, latency 593 usec
Concurrency: 9, throughput: 14163.9 infer/sec, latency 634 usec
Concurrency: 10, throughput: 15385.1 infer/sec, latency 648 usec
Concurrency: 11, throughput: 17159.8 infer/sec, latency 639 usec
Concurrency: 12, throughput: 17865.4 infer/sec, latency 670 usec
Concurrency: 13, throughput: 19425.3 infer/sec, latency 668 usec
Concurrency: 14, throughput: 20567.6 infer/sec, latency 679 usec
Concurrency: 15, throughput: 21360.3 infer/sec, latency 701 usec
Concurrency: 16, throughput: 20463.5 infer/sec, latency 780 usec
```

### Обьяснение выбора оптимизаций
Были опробованы dynamic_batching {}, dynamic_batching: {max_queue_delay_microseconds: 10/100/1000/2000 },
instance_group [{count: 4 ..., max_batch_size разных размеров, но все они так или иначе хуже чем версия без них. Возможно это связано с тем, что сама модель очень простая и поэтому время формирования батчей из приходящий запросов намного больше чем прогон самой модели на батче из 1 элемента.

### Описание .py файлов:
* train.py - обучение модели (ожидаемое поведение для HW2, то есть с логированием результатов обучения в mlflow)
* infer.py - прогон модели на валидационном датасете (ожидаемое поведение файла infer.py для HW1)
* savetooonx.py - сохранение модели в формате .onnx (после выполнения создает файл model.onnx)
* client.py - клиент для работы с тритоном с парой тестов.


### Изменения конифгов:
* Изменение uri mlflow : путь - conf/config.yaml : значение - mlflow.uri
* Изменение портов для Triton Server - triron/docker-compose.yaml
