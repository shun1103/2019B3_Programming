# 萩原研　プログラミング研修

**3日目：tensorflowによる実践演習**
今週は、機械学習で使用されることが多いtensorflowを用いて演習を行っていきます。<br>
今日の流れはざっとこんな感じです。
1. tensorflowの使い方を理解する。
2. 画像処理における代表的な手法のCNNを実装する。
3. 言語処理における代表的手法のLSTMを実装する。

---
## 1. tensorflow
tensorflowを用いると順伝播計算を記述するだけで,誤差逆伝播計算を自動で行ってくれる。<br>
このためプログラマーが行う処理の流れは以下のようになる。
1. 変数定義
2. 順伝播計算の定義
3. 誤差関数の定義
4. 最適化手法の定義
5. セッションの定義（毎回同じ文言を書くだけ）
6. 入力データを1へ入力し,4で定義した最適化手法を実行

例)784次元の入力画像から10次元の出力を得る単層ニューラルネット
1. 入力xと教師データt（外部入力による定数）,
   重み`w`とバイアス`b`（変数）を定義。<br>
   外部入力による定数は`placeholder`で,<br>
   変数は`Variable`で定義する。
```
   x = tf.placeholder(tf.float32, [None, 784])
   t = tf.placeholder(tf.float32, [None, 10])
   w = tf.Variable(tf.zeros([784, 10]))
   b = tf.Variable(tf.zeros([10]))
```
2. 出力`y`に至る計算過程を記述
```
   u = tf.matmul(x,w)+b
   y = tf.nn.softmax(u)
```
3. 教師データ`t`と出力`y`を用いて,誤差関数`loss`を定義（ここではログ交差エントロピーを使用）
```
   loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
```
※  理論的には`loss = -tf.reduce_sum(t * tf.log(y))`で良いのだが<br>
   tensorflowにおいては誤差が小さくなりすぎると`loss=nan`となってしまい正しく学習されなくなる。<br>
   これを回避するために`tf.clip_by_value()`を使用する。

4. 最適化手法train_stepを定義
```
   train_step = tf.train.AdamOptimizer().minimize(loss)
```
5. 以下を書くだけ
```
   sess = tf.InteractiveSession()
   sess.run(tf.initialize_all_variables())
```
6. 4の`train_step`を何度も実行
```
   sess.run(train_step, feed_dict={x: 入力バッチ, t: 教師バッチ})
```
注意点.<p>
   tensorflowによって定義した変数を出力したい場合は下記のようにし,
   `sess.run()`を実行しなくてはならない。ここでは,第1引数の値を求めるのに必要な外部入力を
   `feed_dict={}`によって示す。<br>
   下記の例で,wは外部入力がなくとも値を持つため`feed_dict`は不要<br>
   `y`を計算するには入力データが必要なため,`feed_dict`を用いて`x`に実際の入力を入れる。
```
   sess.run(w)
   sess.run(y, feed_dict={x: 入力バッチ})
```

### 演習3-1. MNISTの分類
784-500-10のニューラルネットを学習してください。
- バッチサイズ（学習時に使用するデータ数）：100
  - 学習に使用するデータは訓練データ60000枚から毎回ランダムに100枚選択
- エポック数（データセットを何周学習させるか）：100
- 重みの初期値：ガウシアン初期化(平均0, 標準偏差0.1)
- バイアスの初期値：0
- SGDの学習係数：0.0001

ヒント.<p>
   入力データに対する正答率を知りたいときは以下の式を（5.）より前に書き`sess.run()`で呼び出す。
```
   correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
   例題では中間層が無いニューラルネットの場合を考えた。<br>
   784-500-10のように中間層を追加するには順伝播計算の部分を書き換える。

## 2. CNNの実装
CNNを用いて画像分類を行う。<br>
CNNモデルの構成は,`入力--(畳み込み✕n--プーリング層)✕m--全結合層--出力`となる。<br>
ここでの全結合層とは演習3-1で作成した多層パーセプトロンと同種のものです。<br>
つまりCNNとは,畳込みとプーリングを行った結果を,多層パーセプトロンの入力とするモノのことなのです。<br>
具体的な処理は演習3-1と同様に以下のようになります。<br>
1. 変数定義
2. 順伝播計算の定義
3. 誤差関数の定義
4. 最適化手法の定義
5. セッションの定義（毎回同じ文言を書くだけ）
6. 入力データを1へ入力し,4で定義した最適化手法を実行

例)784次元の入力画像から10次元の出力を得るCNN(畳み込みフィルタ1枚,プーリング1回)
1. 変数定義
```
   #入力x
   x = tf.placeholder(tf.float32, [None, 784])
   #一次元の入力を画像配列に変形
   x_image = tf.reshape(x, [-1, 28, 28, 1])
```
```
   #畳み込みフィルタを定義
   num_filters = 16
   W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters], stddev=0.1))
```
   ※入力画像サイズが14✕14になっているのは下記のプーリングによって画像サイズが半分になるため。
```
   #全結合層を定義
   num_units1 = 14 * 14 * num_filters
   num_units2 = 1024
   w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
   b2 = tf.Variable(tf.zeros([num_units2]))
   w0 = tf.Variable(tf.zeros([num_units2, 10]))
   b0 = tf.Variable(tf.zeros([10]))
```

2. 順伝播計算の定義
```
   #畳み込み層
   h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='SAME')
```
```
   #プーリング層(stride=[1,2,2,1]より画像サイズが半分になる)
   #stride=[1（固定）,縦ストライド,横ストライド,1（固定）]
   h_pool = tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

```
   ※ストライドとはフィルタを何マスずつ移動させながら畳み込み,プーリングを行うかというもの。
```
   #畳み込み結果を一次元データに変形
   h_pool_flat = tf.reshape(h_pool, [-1, 14 * 14 * num_filters])
```
```
   #全結合層
   hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)
   p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)
   t = tf.placeholder(tf.float32, [None, 10])
```
3. 誤差関数の定義
```
   loss = -tf.reduce_sum(t * tf.log(tf.clip_by_value(p, 1e-10, 1.0)))
```
4. 最適化手法の定義
```
   train_step = tf.train.AdamOptimizer().minimize(loss)
```
5. セッションの定義（毎回同じ文言を書くだけ）
```
   sess = tf.InteractiveSession()
   sess.run(tf.initialize_all_variables())
   saver = tf.train.Saver()
```
6. 入力データを1へ入力し,4で定義した最適化手法を実行
```
   sess.run(train_step, feed_dict={x: 入力バッチ, t: 教師バッチ})
```

### 演習3-2. CNNによるMNISTの分類
MNISTに対し,上記のCNNを学習してください。
- バッチサイズ（学習時に使用するデータ数）：100
  - 学習に使用するデータは訓練データ60000枚から毎回ランダムに100枚選択
- エポック数（データセットを何周学習させるか）：4000

## RNNの実装
RNNを用いて時系列データの学習と予測を行う。
大きく分けて３つの流れで行います。
1. モデルの定義
2. 学習データの作成
3. モデルの利用
使用したライブラリは以下の通りです。
```
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
```

### モデルの定義
モデルの定義は３つの関数を作成します。
- `inference`
- `loss`
- `training`

#### `inference`
この関数はモデル自体の定義をします。
シンプルな擬似コードは下記のようになります。
```
def inference(x):
  s = tanh(matmul(x, U) + matmul(s_prev, W) + b)
  y = matmul(s, V) + c
return y
```
しかしこのままでは`s_prev`が１時刻のみさかのぼることになるので以下のような計算が必要になります。
```
def inference(x, maxlen):
  for t in range(maxlen):
    s[t] = s[t - 1]
  y = matmul(s[t], V) + c
  return y
```
このような時系列に沿った状態を保持しておく`cell`の実装として以下のものが用意されています。
```
cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
```
セルの初期化は以下のように行います。
```
initial_state = cell.zero_state(n_batch, tf.float32)
```
`cell`は現在時刻の入力とセルの内部状態からセルの出力とセルの更新された内部状態を返します。
そのため入力層から出力層の手前までの出力を表す実装は下記のようになります。
```
state = initial_state
outputs = []
with tf.variable_scope('RNN'):
  for t in range(maxlen):
    if t > 0:
      tf.get_variable_scope().reuse_variables()
    (cell_output, state) = cell(x[:, t, :], state)
    outputs.append(cell_output)
output = outputs[-1]
```
`with tf.variable_scope('RNN'):`
と
```
if t > 0:
  tf.get_variable_scope().reuse_variables()
```
は過去の値を取り出すときに過去の変数にアクセスするために必要なものです。

隠れ層から出力層までの実装は下記のとおりです。
```
V = weight_variable([n_hidden, n_out])
c = bias_variable([n_out])
y = tf.matmul(output, V) + c
```

以上をまとめると`inference()`は以下のコードになります。
```
def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)
  def bias_variable(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

  cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
  initial_state = cell.zero_state(n_batch, tf.float32)

  state = initial_state
  outputs = []
  with tf.variable_scope('RNN'):
    for t in range(maxlen):
      if t > 0:
        tf.get_variable_scope().reuse_variables()
      (cell_output, state) = cell(x[:, t, :], state)
      outputs.append(cell_output)
  output = outputs[-1]

  V = weight_variable([n_hidden, n_out])
  c = bias_variable([n_out])
  y = tf.matmul(output, V) + c

  return y
```

#### `loss`
`loss`は２乗平均誤差を用います。
```
def loss(y, t):
  mse = tf.reduce_mean(tf.square(y - t))
  return mse
```

#### `training`
`optimizer`にはAdamを用います。
```
def training(loss):
  optimizer = \
    tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
  train_step = optimizer.minimize(loss)
  return train_step
```

### 学習データの作成
今回はノイズ入りsin波を用います。
これを生成するコードは下記のようになります。
```
def sin(x, T=100):
  return np.sin(2.0 * np.pi * x / T)

def toy_problem(T=100, ampl=0.05):
  x = np.arange(0, 2 * T + 1)
  noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
  return sin(x) + noise
```

データの成形をおこないます。
計算上の都合上１次元の配列を一定の長さで区切ります。
下記の実装になります。
```
length_of_sequences = 2 * T
maxlen = 25

data = []
target = []

for i in range(0, length_of_sequences - maxlen - 1):
  data.append(f[i: i + maxlen])
  target.append(f[i + maxlen])
```

今回のモデルのデータは１次元ですが、RNNに用いるデータのshapeは(データの数、データ長、次元数)である必要があるため以下のようにreshapeします。

```
X = np.array(data).reshape(len(data), maxlen, 1)
Y = np.array(target).reshape(len(data), 1)
```

これでデータの作成が完成しました。
実験のために訓練データと検証データに分割します。
```
N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = \
  train_test_split(X, Y, test_size=N_validation)
```

### モデルの利用
メインの処理で書くモデルの設定に関するコードは下記となります。
```
n_in = len(X[0][0])
n_hidden = 20
n_out = len(Y[0])

x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
t = tf.placeholder(tf.float32, shape=[None, n_out])
n_batch = tf.shape(x)[0]   # 修正：xのshapeから動的にバッチサイズを確保

history = {
  'val_loss': []
}
early_stopping = EarlyStopping(patience=10, verbose=1)

y = inference(x, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
loss = loss(y, t)
train_step = training(loss)
```
また`EarlyStopping`の実装は以下のようになってます。
```
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
```

実際のモデルの学習は以下のようになります。
```
epochs = 500
batch_size = 10

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

n_batches = N_train // batch_size

for epoch in range(epochs):
  X_, Y_ = shuffle(X_train, Y_train)

  for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size

    sess.run(train_step, feed_dict={
      x: X_[start:end],
      t: Y_[start:end],
      # 修正：上の書き方だとエポックの最後でバッチサイズが10じゃなくなりエラーが起きる
      # n_batch: batch_size
    })

  val_loss = loss.eval(session=sess, feed_dict={
    x: X_validation,
    t: Y_validation,
    # n_batch: N_validation
  })
  history['val_loss'].append(val_loss)
  print("epoch:", epoch, ' validation loss:', val_loss)
  if early_stopping.validate(val_loss):
    break
```

これで学習は完了です。
予測誤差の推移は以下のようにプロットすると見れます。
```
plt.rc('font', family='serif')
fig = plt.figure()
ax_loss = fig.add_subplot(111)
ax_loss.plot(range(len(history['val_loss'])), history['val_loss'], label='loss', color='red')
plt.xlabel('epochs')
plt.show()
```

また予測では元データのはじめの長さ（上の例では２５）によって２６時刻目を予測し、またそれを入力に用いて２７時刻目を予測するという流れになります。

```
  truncate = maxlen
  Z = X[:1]
  original = [f[i] for i in range(maxlen)]
  predicted = [None for i in range(maxlen)]

  for i in range(length_of_sequences - maxlen + 1):
      z_ = Z[-1:]
      y_ = y.eval(session=sess, feed_dict={
          x: Z[-1:],
          n_batch: 1
      })
      sequence_ = np.concatenate(
          (z_.reshape(maxlen, n_in)[1:], y_), axis=0) \
          .reshape(1, maxlen, n_in)
      Z = np.append(Z, sequence_, axis=0)
      predicted.append(y_.reshape(-1))

  plt.rc('font', family='serif')
  fig = plt.figure()
  plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='black')
  plt.plot(original, linestyle='dashed', color='blue')
  plt.plot(predicted, color='red')
  plt.show()
```
for文の煩雑な式は出力のサイズを入力のサイズに合わせるために予測値`y_`を加工しているだけです。

## 演習3-3. RNNによるsin波の予測
上記のモデルとデータを用いてsin波の学習と予測を行ってください。
