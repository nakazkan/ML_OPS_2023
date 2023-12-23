# ML_OPS_2023



### Решаемая задача
Классификация ирисов через sklearn - RandomForestClassifier.  
P.S. Почему не торч? Потому что на общажном интеренете для его установки для проверки poetry install в новой среде нужно ждать кучу времени. Если уж очень хочется, чтобы тут обучалась нейронка то можете в фит вставить код, который напишет какая-нибудь Chat-GPT(ну либо perplexity с copilot) на запрос "Write me python code that download MNIST dataset, train the model using PyTorch".

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
* До оптимизаций: 
* После оптимизаций

### Обьяснение выбора оптимизаций
...


### Описание .py файлов:
* train.py - обучение модели (ожидаемое поведение для HW2, то есть с логированием результатов обучения в mlflow)
* infer.py - прогон модели на валидационном датасете (ожидаемое поведение файла infer.py для HW1)
* savetooonx.py - сохранение модели в формате .onnx (после выполнения создает файл model.onnx)
* client.py - клиент для работы с тритоном с парой тестов.


### Изменения конифгов: 
* Изменение uri mlflow : путь - conf/config.yaml : значение - mlflow.uri
* Изменение портов для Triton Server - triron/docker-compose.yaml