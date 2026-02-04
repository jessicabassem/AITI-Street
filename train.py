from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
import numpy as np
import matplotlib.pyplot as plt


# --- 2. إعداد نموذج YOLOE ---
model = YOLOE("yoloe-11s.yaml")

# تحميل الأوزان مسبقًا
model.load("yoloe-11s-seg.pt")

# --- 3. التدريب على بياناتك ---
results = model.train(
    data=r"data\data.yaml",  # Detection dataset
    epochs=100,
    patience=10,
    trainer=YOLOEPETrainer,
)

# --- 4. التقييم ---
metrics = model.val()

# --- 5. رسم Confusion Matrix ---
class_names = ["Car", "Emergency Vehicle"]

# تأكد أنها numpy array
conf_mat = np.array(metrics.confusion_matrix.matrix)
conf_mat = conf_mat[:2, :2]  # قص الصفين والعمودين الأولين

plt.figure(figsize=(6, 5))
plt.imshow(conf_mat, cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(ticks=np.arange(2), labels=class_names)
plt.yticks(ticks=np.arange(2), labels=class_names)

# كتابة الأرقام داخل الخانات
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_mat[i][j], ha='center', va='center', fontsize=12, color='black')

plt.colorbar()
plt.tight_layout()
plt.show()
