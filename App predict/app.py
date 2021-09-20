from flask import Flask, redirect, url_for,render_template, request
import pandas as pd
import numpy as np
import string
import re
import csv

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

import uuid
from datetime import datetime

app = Flask(__name__,template_folder="template")

@app.route("/")
def home():
  return render_template("index.html", content={"data1":"halo dari data 1", "data2":"hello dari data 2"})

@app.route('/predict', methods=('GET', 'POST'))
def predict():
    if request.method == 'POST':
      data = request.files['dataset']
      data_asli = pd.read_csv(data, sep=',')
      nama_file = data.filename.split('.')[0]
      print(nama_file)
      # print(data_asli.head())

      # menghapus duplikasi ulasan
      data_asli=data_asli[~data_asli['review'].duplicated()]
      data_asli=data_asli.reset_index(drop=True)
      # print(len(data_asli))
      # print(data_asli.head())

      data_test=data_asli
      data_tampil=data_asli['review'].tolist()
      # mengubah ke bentuk lower case
      for i in range(len(data_test)):
          review = str.lower(data_test['review'].iloc[i])
          data_test['review'].iloc[i]=review
      # print(data_test.head())

      # menghapus angka
      pattern=r'[0-9]+'
      for i in range(len(data_test)):
          data_test['review'].iloc[i] = re.sub(pattern,' ', data_test['review'].iloc[i], flags=re.MULTILINE)
      # print(data_test.head())

      # menghapus bad karakter dan tanda baca
      pattern=r'[^A-Za-z ]'
      for i in range(len(data_test)):
          data_test['review'].iloc[i] = re.sub(pattern,' ', data_test['review'].iloc[i], flags=re.MULTILINE)
      # print(data_test.head())

      #menghapus spasi ganda
      for i in range(len(data_test)):
          data_test['review'].iloc[i] = re.sub(' +',' ', data_test['review'].iloc[i], flags=re.MULTILINE)
      # print(data_test.head())

      #normalisasi
      reader = csv.reader(open('normalisasi.csv', 'r'))
      d = {}
      for row in reader:
          k,v= row
          d[str.lower(k)] = str.lower(v)
          #print d[k]
      pat = re.compile(r"\b(%s)\b" % "|".join(d))
      for i in range(len(data_test)):
          text = str.lower(data_test['review'].iloc[i])
          text = pat.sub(lambda m: d.get(m.group()), text)
          #print text
          data_test['review'].iloc[i]=text
      # print(data_test.head(10))

      #stemming
      factory = StemmerFactory()
      stemmer = factory.create_stemmer()
      for i in range(len(data_test)):
          sent=data_test['review'].iloc[i]
          output = stemmer.stem(sent)
          data_test['review'].iloc[i]=output
      # print(data_test.head(10))

      #import dataset training
      #untuk membuat daftar vocab
      df = pd.read_csv('dataset_clean_label.csv',header=0, names=['sentimen','text'])
      # print(df.head())

      text = df['text'].tolist()
      # print(len(text))
      
      token = Tokenizer(oov_token='<OOV>')
      token.fit_on_texts(text) #untuk vocab dari data train

      #mengubah data test ke list untuk di prediksi
      test = data_test['review'].tolist()
      print(len(test))
      encode_text = token.texts_to_sequences(test) #sequence dari data test

      max_kata=100
      X = pad_sequences(encode_text, maxlen=max_kata, padding="post")

      opt = SGD(lr=0.008)
      model = load_model('akurasi eks13-80,22 - acc train 84,64.h5')

      model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)

      #predict
      y_pred1 = model.predict(X) #data sequences nya
      y_pred = np.argmax(y_pred1, axis=1)
      print(y_pred)

      #menghitung jumlah per label
      pos=np.count_nonzero(y_pred == 1)
      print(pos)
      net=np.count_nonzero(y_pred == 0)
      print(net)
      neg=np.count_nonzero(y_pred == 2)
      print(neg)
      
      y_pred = np.array(y_pred).astype('str').tolist()
      y_pred = ['Positif' if i=='1' else 'Negatif' if i=='2' else "Netral" for i in y_pred]
      
      my_dict = {"sentimen": y_pred, "review": data_tampil}
      df=pd.DataFrame(my_dict)
      
      akses = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')
      unique_filename = str(uuid.uuid4())
      df.to_csv("output/"+nama_file+"-"+akses+".csv")

      return render_template('hasil_predict.html', pos=pos, net=net, neg=neg, dataTmpl_hslPred=zip(data_tampil,y_pred))

    return render_template('hasil_predict.html')




@app.route("/<name>")
def user(name):
  return f"Hello {name}!"

@app.route("/admin")
def admin():
	return redirect(url_for("home"))

@app.route('/tesform', methods=('GET', 'POST'))
def tesform():
  if request.method == 'POST':
    username = request.form['username']
    password = request.form['password']
    error = None # inisialisasi var eror 

    if not username:
      error = 'Username is required.'
    elif not password:
      error = 'Password is required.'

    if error is None:
      # return redirect(url_for('home'))
      return render_template('tes.html', content=username+" "+password)
    else:
      print("ERROR") # ngeprint di console

  return render_template('tes.html')

if __name__ == "__main__":
  app.run(debug=True)