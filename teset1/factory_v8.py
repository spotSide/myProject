# color detector 검출하기위한 뼈대
import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
import cv2
import numpy as np
import openvino as ov
from openvino.runtime import Core
from iotdemo import FactoryController, MotionDetector, ColorDetector

FORCE_STOP = False

def thread_cam1(q):
    # MotionDetector 초기화
    motion_detector = MotionDetector()
    motion_detector.load_preset('/home/intel/git-training/DX-01/class02/smart-factory/resources/motion.cfg')
    
    # OpenVINO Core 및 모델 초기화
    core = ov.Core()
    model = core.read_model('/home/intel/workdir1/otx-workspace/otx-workspace/20241206_070526/exported_model.xml')  # 모델 경로를 실제 경로로 변경 필요
    compiled_model = core.compile_model(model, 'CPU')  # 'CPU'를 다른 디바이스로 변경 가능

    # 모델 입출력 레이어 설정
    input_layer = next(iter(compiled_model.inputs))
    output_layer = next(iter(compiled_model.outputs))

    # 클래스 ID와 이름 매핑
    class_names = {0: "Circle", 1: "Cross"}  # ID 0은 Circle, ID 1은 Cross를 나타냄

    # 동영상 파일 열기
    cap = cv2.VideoCapture('/home/intel/git-training/DX-01/class02/smart-factory/resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 프레임 큐에 삽입
        q.put(("VIDEO:Cam1 live", frame))

        # 모션 감지
        detected = motion_detector.detect(frame)
        if detected is not None:
            q.put(("VIDEO:Cam1 detected", detected))

        # OpenVINO 추론 수행
            resized_frame = cv2.resize(frame, (input_layer.shape[2], input_layer.shape[3]))
            input_tensor = np.expand_dims(resized_frame.transpose(2, 0, 1), axis=0)
            results = compiled_model.infer_new_request({input_layer.any_name: input_tensor})
            predictions = results[output_layer]

            # 가장 높은 확률의 클래스 ID와 확률 출력
            top_class = np.argmax(predictions)
            # todo confidence가 o x 나와야 하지 않은가
            # 딱히 구분하고 있지 않음
            confidence = predictions[0][top_class]
            class_name = class_names.get(top_class, "Unknown")  # 클래스 이름 가져오기

            # 결과 큐에 삽입
            q.put(("RESULT:Cam1", f"Detected: {class_name}, Confidence: {confidence:.2f}"))

            # 프레임에 결과 시각화
            cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cap.release()
    q.put(('DONE', None))

def thread_cam2(q):
    # MotionDetector 초기화
    motion_detector = MotionDetector()
    motion_detector.load_preset('/home/intel/git-training/DX-01/class02/smart-factory/resources/motion.cfg')
    
    # ColorDetector 초기화
    color_detector = ColorDetector()
    color_detector.load_preset('/home/intel/git-training/DX-01/class02/smart-factory/resources/color.cfg')

    # 동영상 파일 열기
    cap = cv2.VideoCapture('/home/intel/git-training/DX-01/class02/smart-factory/resources/conveyor.mp4')

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 프레임 큐에 삽입
        q.put(("VIDEO:Cam2 live", frame))

        # 모션 감지
        detected = motion_detector.detect(frame)
        if detected is not None:
            q.put(("VIDEO:Cam2 detected", detected))

            # 색상 감지 수행
            predict = color_detector.detect(detected)

            # 확률 추출
            blue, white = predict  # blue = (ID, confidence), white = (ID, confidence)
            blue_confidence = blue[1]
            white_confidence = white[1]

            # 가장 높은 확률의 색상 결정
            if blue_confidence > white_confidence:
                color_result = "Blue"
                confidence = blue_confidence
            else:
                color_result = "White"
                confidence = white_confidence

            # 결과를 큐에 삽입
            q.put(("RESULT:Cam2", f"Detected Color: {color_result}, Confidence: {confidence:.2f}"))

            # 프레임에 결과 시각화
            cv2.putText(frame, f"{color_result}: {confidence:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 큐에 시각화용 프레임 추가
            q.put(("VIDEO:Cam2 result", frame))

    cap.release()
    q.put(('DONE', None))



def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)

def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # 큐 생성
    q = Queue()

    # 스레드 생성 및 시작
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        try:
            while not FORCE_STOP:
                if cv2.waitKey(10) & 0xff == ord('q'):
                    FORCE_STOP = True
                    break

                try:
                    # 큐에서 항목 가져오기
                    name, data = q.get(timeout=0.1)

                    if name.startswith("VIDEO:"):
                        imshow(name[6:], data)
                    if name.startswith("RESULT:"):
                        print(name, data)

                    # 액추에이터 제어
                    if name == 'PUSH':
                        ctrl.push_actuator(data)

                    if name == 'DONE':
                        FORCE_STOP = True

                    q.task_done()
                    
                except Empty:
                    continue

        finally:
            cv2.destroyAllWindows()
            t1.join()
            t2.join()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        os._exit(1)
