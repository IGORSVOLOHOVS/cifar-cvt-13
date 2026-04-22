Классификация CIFAR
Теперь, когда вы научились отличать CvT от ViT, пора порешать задачи с помощью CvT. 
Рассмотрим задачу классификации на датасете CIFAR. Это альтернативный ImageNet датасет, который помогает оценивать точность моделей и содержит 60000 изображений, разделённых на 10 классов. Так как CvT по умолчанию предобучены на ImageNet, их выходной слой предусматривает классификацию на 1000 классов ImageNet. Поэтому нужно корректировать выходной слой CvT для решения задачи классификации на CIFAR.
Код загрузки CvT из библиотеки transformers похож на код загрузки Swin. 
Пример microsoft/cvt-13 — стандартной версии CvT из оригинальной статьи:

from transformers import AutoFeatureExtractor, CvtForImageClassification

# Выбор предобученной модели
model_name = "microsoft/cvt-13"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = CvtForImageClassification.from_pretrained(model_name) 
Различие только в том, что вместо Auto Image Processor используется Auto Feature Extractor (извлечение признаков), так как, в отличие от Swin и ViT, CvT начинается с операции свёртки. 
Для использования в датасетах Feature Extractor так же, как и предобработчик ViT, нужно обернуть во вспомогательную функцию:

def cvt_transform(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0) 
Для скачивания CIFAR и добавления предобработки используются команды:

from torchvision import datasets

train_dataset = datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=cvt_transform
)
test_dataset = datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=cvt_transform
) 
Так как в этом датасете по умолчанию не предусмотрен валидационный датасет, его нужно сделать самостоятельно, разделив обучающий на две части с помощью функции random_split. В примере ниже для валидационного датасета выбрали 5000 образцов из 50000:

from torch.utils.data import DataLoader, random_split

val_size = 5000
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size]) 
Для замены последнего слоя CvT на классификатор используется такая же операция, как и для ViT:

model.classifier = nn.Linear(
    model.classifier.in_features, len(train_dataset.dataset.classes)
) 
Обратите внимание, что для получения классов из обучающего датасета нужно вызвать не .classes, а .dataset.classes — поскольку обучающий датасет разделили на две части, теперь он является объектом типа Subset, а не Dataset, как прежде.
Процесс дообучения аналогичен ViT. Выполните его в практическом задании.
Задание 1
Для выполнения этого задания подключитесь к виртуальной машине.
Дообучите CvT "microsoft/cvt-13" из библиотеки transformers для классификации датасета CIFAR. Проверьте точность распознавания обученной модели на тестовом датасете CIFAR.
Обученную модель поместите в переменную model, тестовый датасет — в test_dataset, а загрузчик для него —в test_loader. Это необходимо для корректного расчёта точности.
Обучайте на GPU. Убедитесь, что модель и все данные, с которыми она работает, переведены на GPU c помощью метода .to() :

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = inputs.to(device)
labels = labels.to(device) 
Дообучить достаточно на одной эпохе. Если задание выполнено верно, общая точность распознавания превысит 65%.
Прекод

# ВАШ КОД ДЛЯ ЗАГРУЗКИ CIFAR, CvT и её дообучения
# Подсчет точности
from sklearn.metrics import classification_report

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels_predicted = []
labels_true = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)  # argmax по всем примерам
        labels_predicted.extend(predicted.cpu().numpy())
        labels_true.extend(labels.cpu().numpy())

print(
    classification_report(
        labels_true, labels_predicted, target_names=test_dataset.classes
    )
) 


Подсказка