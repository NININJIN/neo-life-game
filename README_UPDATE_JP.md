# Neo Life Game v31 Cloud Stable

Streamlit Cloudで落ちていた `label got an empty value` と `use_container_width` 警告を避けるための安定版です。

## 反映方法
1. このフォルダ内の `app.py` を、GitHubリポジトリ直下の `app.py` と置き換える。
2. `runtime.txt` と `requirements.txt` もリポジトリ直下に置く。
3. `.streamlit/config.toml` も必要なら置き換える。
4. GitHub Desktopで commit → push。
5. Streamlit Cloudで再起動、またはページを再読み込み。

## 注意
`runtime.txt` は Python 3.11 を指定します。Cloudが Python 3.14 を選ぶことで起きる互換性問題を避けるためです。
