"""ネオライフゲーム共有用QRコード作成スクリプト。
使い方:
    python make_qr.py "https://あなたのアプリURL"
出力:
    neo_life_qr.png
"""
import sys
from pathlib import Path

import qrcode


def main() -> None:
    if len(sys.argv) >= 2:
        url = sys.argv[1].strip()
    else:
        url = input("公開URLを貼ってください: ").strip()

    if not (url.startswith("http://") or url.startswith("https://")):
        raise SystemExit("URLは http:// または https:// から始めてください。")

    img = qrcode.make(url)
    out = Path("neo_life_qr.png")
    img.save(out)
    print(f"QRコードを作成しました: {out.resolve()}")


if __name__ == "__main__":
    main()
