# v10 영상 심도 0.0~1.0 까지 받아서 섹션을 16등분하고 hd해상도로 출력하기
import cv2
import numpy as np
import time
import requests
from pathlib import Path
import openvino as ov
import openvino.properties as props
from notebook_utils import download_file

# notebook_utils.py 다운로드 (필요 시)
if not Path("notebook_utils.py").exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    with open("notebook_utils.py", "w") as f:
        f.write(r.text)

# 모델 다운로드 및 설정
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

# WebcamProcessor 클래스
class WebcamProcessor:
    def __init__(self, camera_id=0, frame_width=1280, frame_height=720):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError("웹캠을 열 수 없습니다.")
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.current_frame = None

    def read_frame(self):
        """웹캠으로부터 프레임을 읽어옵니다."""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("웹캠에서 영상을 읽을 수 없습니다.")
        self.current_frame = frame
        return frame

    def release(self):
        """웹캠 자원을 해제합니다."""
        self.cap.release()

# DepthProcessor 클래스
class DepthProcessor:
    def __init__(self, compiled_model, input_key, output_key):
        self.compiled_model = compiled_model
        self.input_key = input_key
        self.output_key = output_key

    def process_frame(self, frame):
        """주어진 프레임에서 뎁스 결과를 생성합니다."""
        resized_frame = cv2.resize(frame, (self.input_key.shape[2], self.input_key.shape[3]))
        input_image = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
        result = self.compiled_model([input_image])[self.output_key]
        return result

    def visualize_result(self, result):
        """뎁스 결과를 시각화합니다."""
        result_frame = self.convert_result_to_image(result)
        return result_frame

    @staticmethod
    def normalize_minmax(data):
        """뎁스 데이터를 정규화합니다."""
        return (data - data.min()) / (data.max() - data.min())

    def convert_result_to_image(self, result, colormap="viridis"):
        """뎁스 결과를 컬러맵으로 변환합니다."""
        import matplotlib.cm
        cmap = matplotlib.cm.get_cmap(colormap)
        result = result.squeeze(0)
        result = self.normalize_minmax(result)
        result = cmap(result)[:, :, :3] * 255
        result = result.astype(np.uint8)
        return result


def process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8):
    """
    16등분 섹션의 대표 심도 값을 계산하고, 기준값 이상의 섹션을 왼쪽과 오른쪽으로 나눕니다.
    
    Args:
        depth_map: 정규화된 뎁스 맵 (0.0 ~ 1.0)
        num_rows: 세로 섹션 수
        num_cols: 가로 섹션 수
        threshold: 기준 심도 값
    
    Returns:
        decision: "Avoid to Right", "Avoid to Left", or "Balanced"
    """
    h, w = depth_map.shape
    section_height = h // num_rows
    section_width = w // num_cols
    
    left_count = 0
    right_count = 0

    for row in range(num_rows):
        for col in range(num_cols):
            # 섹션 좌표 계산
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width

            # 섹션 영역 추출
            section = depth_map[y1:y2, x1:x2]
            
            # 섹션의 평균 심도 계산
            mean_depth = section.mean()
            
            # 기준값 이상인지 확인
            if mean_depth >= threshold:
                if col < num_cols // 2:
                    left_count += 1  # 왼쪽 영역
                else:
                    right_count += 1  # 오른쪽 영역

    # 결정 논리
    if left_count > right_count:
        return "Avoid to Right"
    elif right_count > left_count:
        return "Avoid to Left"
    else:
        return "Balanced"
    
def display_depth_sections(image, depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720):
    """
    화면을 num_rows x num_cols 섹션으로 나누고, 각 섹션의 대표 심도 값을 계산하여 출력 크기에 맞춰 표시합니다.
    
    Args:
        image: 원본 프레임 (BGR 이미지)
        depth_map: 정규화된 뎁스 맵 (0.0 ~ 1.0)
        num_rows: 세로 섹션 수
        num_cols: 가로 섹션 수
        output_width: 출력 화면의 폭
        output_height: 출력 화면의 높이
    """
    # 출력 해상도로 크기 조정
    image = cv2.resize(image, (output_width, output_height))
    depth_map = cv2.resize(depth_map, (output_width, output_height))

    # 섹션 크기 계산
    section_height = output_height // num_rows
    section_width = output_width // num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            # 섹션 좌표 계산
            y1, y2 = row * section_height, (row + 1) * section_height
            x1, x2 = col * section_width, (col + 1) * section_width

            # 섹션 영역 추출
            section = depth_map[y1:y2, x1:x2]
            
            # 섹션의 평균 심도 계산
            mean_depth = section.mean()

            # 섹션에 심도 값 표시
            cv2.putText(
                image,
                f"{mean_depth:.2f}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 흰색 텍스트
                1,
                cv2.LINE_AA
            )
            
            # 섹션 경계선 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 초록색 경계선

    return image

# 메인 루프
def main():
    # 웹캠 및 모델 초기화
    webcam = WebcamProcessor()
    
    # OpenVINO 모델 초기화
    core = ov.Core()
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)
    core.set_property({props.cache_dir(): cache_folder})
    model = core.read_model(model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="GPU.1")
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    
    depth_processor = DepthProcessor(compiled_model, input_key, output_key)

    try:
        while True:
            frame = webcam.read_frame()  # 웹캠에서 프레임 읽기
            depth_result = depth_processor.process_frame(frame)  # 뎁스 처리
            
            # 뎁스 맵 정규화
            depth_map = (depth_result.squeeze(0) - depth_result.min()) / (depth_result.max() - depth_result.min())
            
            # 16등분 섹션 처리 및 결정
            decision = process_depth_sections(depth_map, num_rows=4, num_cols=4, threshold=0.8)
            
            # 시각화된 뎁스 맵
            depth_frame = depth_processor.visualize_result(depth_result)
            
            # 16등분 영역에 심도 정보 표시 (1280x720 크기로 출력)
            display_frame = display_depth_sections(depth_frame.copy(), depth_map, num_rows=4, num_cols=4, output_width=1280, output_height=720)
            
            # 결정 결과 출력
            cv2.putText(
                display_frame,
                decision,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 빨간색 텍스트
                2,
                cv2.LINE_AA
            )
            
            # 실시간 출력
            cv2.imshow("Depth Estimation with Sections", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
                break
    except KeyboardInterrupt:
        print("처리가 중단되었습니다.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
