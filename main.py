from model import *
from data import *
# from keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                                  allow_soft_placement=True, device_count={'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def draw_history(history):
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(data_set + 'results/Model accuracy.png')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.savefig(data_set + 'results/Model loss.png')
    plt.show()


data_set = 'data/' + 'ocean_sub1/'  # 'membrane/' 'ocean_sub1/' 'ocean/'

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, data_set + 'train', 'image', 'label', data_gen_args, save_to_dir=None)
model = unet()
# plot_model(model, to_file='results/model_structure.png')
model_checkpoint = ModelCheckpoint(data_set + 'results/unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit(myGene, steps_per_epoch=10, epochs=10, callbacks=[model_checkpoint])  # 300
draw_history(history)

model = load_model(data_set + 'results/unet.hdf5')
testGene = testGenerator(data_set + 'test', num_image=30)
results = model.predict(testGene, steps=30, verbose=1)
results_path = data_set + "results/prediction"
if not os.path.exists(results_path):
    os.makedirs(results_path)
saveResult(results_path, results)
