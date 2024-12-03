import requests
import time
import cv2
import numpy as np
from pathlib import Path
import openvino as ov
import openvino.properties as props
from notebook_utils import download_file, load_image

# Fetch the 'notebook_utils' module if not available
if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    with open("notebook_utils.py", "w") as f:
        f.write(r.text)

# Model preparation
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)

ir_model_url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
ir_model_name_xml = "MiDaS_small.xml"
ir_model_name_bin = "MiDaS_small.bin"

if not (model_folder / ir_model_name_xml).exists():
    download_file(ir_model_url + ir_model_name_xml, filename=ir_model_name_xml, directory=model_folder)
if not (model_folder / ir_model_name_bin).exists():
    download_file(ir_model_url + ir_model_name_bin, filename=ir_model_name_bin, directory=model_folder)

model_xml_path = model_folder / ir_model_name_xml

# Helper functions
def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1."""
    return (data - data.min()) / (data.max() - data.min())

def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    import matplotlib.cm
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result

def to_rgb(image_data) -> np.ndarray:
    """Convert image_data from BGR to RGB."""
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)


# 기존 설정 부분은 동일

# Set up OpenVINO Core, compile model, and configure device
core = ov.Core()
cache_folder = Path("cache")
cache_folder.mkdir(exist_ok=True)
core.set_property({props.cache_dir(): cache_folder})
model = core.read_model(model_xml_path)
device = "GPU.1"  # Assuming CPU is used; you can change this if needed
compiled_model = core.compile_model(model=model, device_name=device)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 0번 웹캠 사용 (다른 번호의 웹캠은 1, 2 등으로 변경)
if not cap.isOpened():
    raise ValueError("웹캠을 열 수 없습니다.")

# 출력 설정 (출력 화면 사이즈 조절하기)
target_frame_height = 720  # 원하는 출력 해상도 (예: 480p)
target_frame_width = 1280
SCALE_OUTPUT = 0.5  # 영상 축소 비율
FOURCC = cv2.VideoWriter_fourcc(*"vp09")
output_directory = Path("output")
output_directory.mkdir(exist_ok=True)
result_video_path = output_directory / "webcam_monodepth.mp4"

out_video = cv2.VideoWriter(
    str(result_video_path),
    FOURCC,
    1,  # 초당 30 프레임으로 저장  / 초당프레임 (소숫점도 인식함)
    (target_frame_width * 2, target_frame_height),  # 원본+결과 병합 프레임 크기
)

start_time = time.perf_counter()
total_inference_duration = 0

# 특정 BGR 범위 설정
lower_bound = np.array([100, 50, 50])  # 최소 BGR 값
upper_bound = np.array([140, 255, 255])  # 최대 BGR 값

try:
    while True:
        ret, image = cap.read()
        if not ret:
            print("웹캠에서 영상을 읽을 수 없습니다.")
            break

        # 네트워크 입력 크기로 변경
        resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        # 모델 추론
        inference_start_time = time.perf_counter()
        result = compiled_model([input_image])[output_key]
        inference_stop_time = time.perf_counter()
        inference_duration = inference_stop_time - inference_start_time
        total_inference_duration += inference_duration

        # 결과 변환 및 시각화
        result_frame = to_rgb(convert_result_to_image(result))
        result_frame = cv2.resize(result_frame, (target_frame_width, target_frame_height))

        # 9등분 영역 계산
        h, w = result_frame.shape[:2]
        h_step, w_step = h // 3, w // 3

        # 1 | 2 | 3
        # ---------
        # 4 | 5 | 6
        # ---------
        # 7 | 8 | 9

        # 각 섹션의 대표 BGR 값 계산
        section_bgr_values = []
        for i in range(3):  # 세로 방향
            for j in range(3):  # 가로 방향
                y1, y2 = i * h_step, (i + 1) * h_step
                x1, x2 = j * w_step, (j + 1) * w_step
                section = result_frame[y1:y2, x1:x2]
                mean_bgr = cv2.mean(section)[:3]  # 평균 BGR 값 계산
                section_bgr_values.append(mean_bgr)

                # 화면에 섹션 정보 표시
                text = f"BGR: ({int(mean_bgr[0])}, {int(mean_bgr[1])}, {int(mean_bgr[2])})"
                cv2.putText(
                    result_frame,
                    text,
                    (x1 + 10, y1 + 30),  # 텍스트 위치
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # 폰트 크기
                    (255, 255, 255),  # 흰색 텍스트
                    1,  # 텍스트 두께
                    cv2.LINE_AA,
                )

        # 원본과 심도 결과 병합
        image = cv2.resize(image, (target_frame_width, target_frame_height))
        stacked_frame = np.hstack((image, result_frame))
        out_video.write(stacked_frame)

        # 실시간 화면 출력
        cv2.imshow('Monodepth with Extracted Color and BGR Info', stacked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

except KeyboardInterrupt:
    print("처리가 중단되었습니다.")
finally:
    cap.release()
    out_video.release()
    duration = time.perf_counter() - start_time
    print(f"총 {duration:.2f}초 동안 웹캠 데이터를 처리했습니다.")

cv2.destroyAllWindows()