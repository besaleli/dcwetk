from encoder.encoder import encoder
from flask import Flask, request

app = Flask(__name__)
config = {'modelName': 'onlplab/alephbert-base', 'tokenizerName': 'onlplab/alephbert-base'}
alephBert = encoder(config)


@app.route('/encode', methods=['POST'])
def embed_text():
    data = request.get_json()
    df, msg = alephBert.embed(data['tokens'])
    print(msg)
    print(df)
    return df.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
