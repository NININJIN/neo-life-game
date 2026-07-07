# ネオライフゲーム QR共有・公開手順

このフォルダは、ネオライフゲームを他人のスマホから開けるようにするための公開用セットです。

## 中身

- `app.py`：ネオライフゲーム本体
- `requirements.txt`：必要ライブラリ
- `.streamlit/config.toml`：Streamlit設定
- `make_qr.py`：公開URLからQRコード画像を作るスクリプト
- `run_local.bat` / `run_local.sh`：ローカル起動用
- `Dockerfile`：Docker対応サービス用

## 推奨：Streamlit Community Cloudで公開する

1. GitHubで新しいリポジトリを作る。
2. このフォルダの中身をアップロードする。
3. Streamlit Community Cloudを開く。
4. `Create app` を押す。
5. GitHubリポジトリ、ブランチ、`app.py` を選ぶ。
6. `Deploy` を押す。
7. `https://...streamlit.app` のようなURLができる。

## QRコードを作る

公開URLができたら、PCでこのフォルダを開いて以下を実行します。

```bash
python make_qr.py "https://あなたのアプリURL"
```

`neo_life_qr.png` ができます。これを印刷・LINE送信・スライド貼り付けすれば、スマホで読み取れます。

## すぐ見せたい場合：ローカル + ngrok

PCでアプリを起動します。

```bash
python -m streamlit run app.py --server.port 8501
```

別のターミナルで、ngrokを入れている場合は次を実行します。

```bash
ngrok http 8501
```

表示された `https://...ngrok-free.app` のようなURLを `make_qr.py` に渡すとQR化できます。
この方法はPCを閉じると止まります。常設公開にはStreamlit Community Cloudなどを使ってください。

## 注意

- 公開URLを知っている人は誰でも触れます。個人情報は入れないでください。
- 多人数が同時に重い設定で動かすと遅くなる場合があります。
- スマホでは「表示プリセット：軽量」「更新間隔短め」「盤面80×120以下」がおすすめです。
- 各閲覧者の操作状態は基本的に別セッションですが、ページを再読み込みすると状態が戻ることがあります。
