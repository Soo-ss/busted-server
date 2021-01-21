import datetime
import cv2
import boto3
import botocore
from flask import Flask, jsonify

app = Flask(__name__)
bucket_name = "busted"

# 캡처부분
def capture():
    capture = cv2.VideoCapture("./road.mp4")

    width = int(capture.get(3))  # 가로
    height = int(capture.get(4))  # 세로

    while capture.isOpened:
        ret, frame = capture.read()
        if ret == False:
            break

        # cv2.imshow("VideoFrame", frame)

        now = datetime.datetime.now().strftime("%d_%H-%M-%S")
        key = cv2.waitKey(33)  # 1) & 0xFF

        cv2.IMREAD_UNCHANGED
        cv2.imwrite("./capture/test" + ".png", frame)

    capture.release()
    cv2.destroyAllWindows()


def uploadS3():
    # 업로드하고자 하는 파일 명
    in_file = "test.png"

    # S3에 다 올려졌을떄의 파일 명
    out_file = in_file

    s3 = boto3.client("s3")
    s3.upload_file("./capture/" + in_file, bucket_name, "images/" + out_file)


# 이건 쓸일 없을듯
def downloadS3():
    # S3에 존재하는 파일명
    in_file = "test.png"
    out_file = in_file

    s3 = boto3.resource("s3")

    try:
        s3.Bucket(bucket_name).download_file("images/" + in_file, out_file)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("해당 파일이 S3에 없습니다.")
        else:
            raise


# uploadS3()

# ================== 여기는 Flask서버 부분이지만 우리는 무한루프 돌릴거라서 일단은 안쓸듯해요 ========
@app.route("/")
def index():
    # capture()
    return "Hello flask"


@app.route("/api/user")
def hello():
    url = "http://localhost:5000/capture/test.png"
    justTest = {"hello": "this is word", "url": url}
    return jsonify(justTest)
