from pathlib import Path

from ultralytics import YOLOWorld


def main():
    image_path = Path("data/mydesk.JPG")

    model = YOLOWorld("yolov8x-worldv2.pt")  # Load a YOLOWorld model

    model.set_classes(["key", "watch"])
    
    results = model.predict(
        source=image_path
    )

    # 結果を画像ファイルとして保存（ヘッドレス環境ではshow()が使えないため）
    results[0].save("output.jpg")
    print("Result saved to output.jpg")


if __name__ == "__main__":
    main()
