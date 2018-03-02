
# ex4.py
# MNISTの画像分類

# データセット読み込み用
from dataset.mnist import load_mnist
from NeuralNetwork import NeuralNetwork
import numpy as np

def main():
    mnist = load_mnist()
    model = NeuralNetwork()

    batch_size = 100
    train_images = 60000
    test_images = 10000
    train_epochs = 100
    train_iters = train_epochs * (train_images // batch_size)

    for i in range(train_iters):
        # 60,000の数字の中からランダムに100個の数字を選ぶ
        indices = np.random.choice(train_images, batch_size)
        # ミニバッチの獲得
        minibatch_image = mnist["train_img"][indices]
        minibatch_label = mnist["train_label"][indices]
        model.sgd(minibatch_image, minibatch_label)
        # 100ループごとに損失関数を表示
        if i % 100 == 0:
            print("Loss {0}: {1}".format(
                i, model.loss(minibatch_image, minibatch_label)))
            # 正解率を表示
            accuracy = np.average(
                (np.argmax(minibatch_label, axis=1) == \
                 np.argmax(model.forward(minibatch_image), axis=1)).astype(int))
            print("Acc: {0} %".format(accuracy * 100))

    # テストデータに対する損失を表示
    print("Test Loss: {0}".format(model.loss(
        mnist["test_img"], mnist["test_label"])))
    test_acc = np.average(
        (np.argmax(mnist["test_label"], axis=1) == \
         np.argmax(model.forward(mnist["test_img"]), axis=1)).astype(int))
    print("Test Acc: {0} %".format(test_acc * 100))

if __name__ == "__main__":
    main()
