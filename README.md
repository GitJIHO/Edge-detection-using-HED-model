# HED(Holistically-Nested Edge Detection) 모델을 활용한 에지 검출 구현 보고서

## 1. 개요

본 프로젝트는 CVPR 2015에서 발표된 "Holistically-Nested Edge Detection(HED)" 논문에 소개된 딥러닝 기반 에지 검출 모델을 구현하고 얼굴 이미지에 적용한 결과를 소개합니다. HED는 전통적인 에지 검출 방법(예: Canny)보다 더 자연스러운 에지를 추출할 수 있는 딥러닝 기반 방식으로, 사전 학습된 모델을 활용하여 얼굴 이미지의 경계선을 정확하게 감지합니다.

## 2. 이론적 배경

### HED(Holistically-Nested Edge Detection) 알고리즘

HED는 VGG16 네트워크 구조를 기반으로 한 Fully Convolutional Neural Network(FCN)입니다. 이 모델의 주요 특징은 다음과 같습니다:

1. **다중 스케일 특징 학습**: 네트워크의 각 단계에서 서로 다른 스케일의 특징을 추출하여 다양한 수준의 에지 정보를 포착합니다.
2. **깊은 감독(Deep Supervision)**: 네트워크의 여러 단계에서 중간 출력을 학습하고 이를 결합하는 방식을 사용합니다.
3. **홀리스틱 에지 검출**: 픽셀 단위로 독립적인 분류가 아닌, 이미지 전체를 고려한 에지 검출을 수행합니다.

기존의 Canny와 같은 전통적인 에지 검출 알고리즘은 저수준 특징(gradient 정보)에만 의존하지만, HED는 다양한 수준의 특징을 학습하여 더 의미론적이고 자연스러운 에지를 추출할 수 있습니다.

## 3. 구현 코드 설명

이 프로젝트에서는 OpenCV의 DNN 모듈을 활용하여 사전 학습된 HED 모델을 불러오고 추론을 수행합니다. 주요 구성 요소는 다음과 같습니다:

### 3.1 CropLayer 클래스

```python
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]
```

이 클래스는 OpenCV의 DNN 모듈에서 HED 모델을 로드할 때 필요한 커스텀 레이어입니다. 모델의 아키텍처에서 필요로 하는 크롭 연산을 처리합니다. 네트워크 내부의 특징맵 크기를 조정하여 다양한 층의 출력을 결합할 수 있게 해줍니다.

### 3.2 모델 다운로드 함수

```python
def download_hed_model():
    """HED 모델 파일들을 다운로드합니다."""

    # deploy.prototxt 다운로드
    prototxt_url = "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
    prototxt_path = "deploy.prototxt"

    if not os.path.exists(prototxt_path):
        print("deploy.prototxt 다운로드 중...")
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        print("deploy.prototxt 다운로드 완료!")

    # 사전 훈련된 모델 다운로드
    model_url = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
    model_path = "hed_pretrained_bsds.caffemodel"

    if not os.path.exists(model_path):
        print("HED 사전 훈련 모델 다운로드 중... (약 56MB)")
        urllib.request.urlretrieve(model_url, model_path)
        print("모델 다운로드 완료!")

    return prototxt_path, model_path
```

이 함수는 GitHub에서 모델 구조 파일(deploy.prototxt)과 BSDS 데이터셋에서 사전 학습된 가중치 파일(hed_pretrained_bsds.caffemodel)을 다운로드합니다.

### 3.3 에지 검출 함수

```python
def hed_edge_detection(image_path, width=500, height=500):
    # 모델 파일 다운로드
    prototxt_path, model_path = download_hed_model()

    # 네트워크 로드
    net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

    # 이미지 읽기 및 전처리
    image = cv.imread(image_path)
    original_height, original_width = image.shape[:2]
    image_resized = cv.resize(image, (width, height))
    
    # 블롭 생성 (HED 모델에 맞는 전처리)
    blob = cv.dnn.blobFromImage(image_resized,
                               scalefactor=1.0,
                               size=(width, height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False,
                               crop=False)

    # 네트워크에 입력 설정 및 추론
    net.setInput(blob)
    output = net.forward()

    # 출력 후처리
    edge_map = output[0, 0]
    edge_map = cv.resize(edge_map, (original_width, original_height))
    edge_map = (255 * edge_map).astype(np.uint8)

    return image, edge_map
```

이 함수는 HED 모델을 사용하여 입력 이미지에서 에지를 검출합니다. 주요 단계는 다음과 같습니다:
1. 모델 파일 다운로드 및 네트워크 로드
2. 이미지 전처리(리사이징)
3. 딥러닝 모델의 입력 형식으로 변환(blob 생성)
4. 네트워크를 통한 추론 수행
5. 출력 후처리(원본 크기로 리사이징 및 정규화)

### 3.4 결과 시각화 함수

```python
def visualize_results(original_image, edge_map, canny_comparison=True):
    if canny_comparison:
        # Canny 엣지 검출 (비교용)
        gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
        canny_edges = cv.Canny(gray, 50, 150)

        # 4개 서브플롯으로 비교 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 원본, HED 결과, Canny 결과, 오버레이 이미지 표시
        # ...
```

이 함수는 원본 이미지와 에지 검출 결과를 시각화합니다. 추가적으로 전통적인 Canny 에지 검출기와의 비교를 위해 동일한 이미지에 Canny 알고리즘을 적용한 결과도 함께 표시합니다.

## 4. 코드 분석

### 4.1 주요 매개변수 선택

1. **이미지 전처리 매개변수**:
   - 입력 이미지를 500×500 크기로 리사이징: 모델의 입력 크기에 맞추고 계산 효율성을 높입니다.
   - 평균값 빼기 (104.00698793, 116.66876762, 122.67891434): 이는 ImageNet 데이터셋의 BGR 평균값으로, 모델 학습 시 사용된 전처리 방식을 동일하게 적용합니다.

2. **Canny 에지 검출 매개변수**:
   - 임계값 (50, 150): 비교를 위한 Canny 알고리즘의 하한 및 상한 임계값으로, 일반적인 에지 검출에 적합한 값입니다.

3. **시각화 매개변수**:
   - 오버레이 가중치 (0.7, 0.3): 원본 이미지와 에지 맵을 결합할 때 각각의 가중치로, 에지가 원본 이미지 위에 잘 보이도록 조정되었습니다.

### 4.2 HED와 Canny 비교 분석

HED 모델은 Canny 알고리즘과 달리 다음과 같은 장점을 갖습니다:
- 다양한 스케일의 특징을 추출하여 더 자연스러운 에지 검출
- 노이즈에 강한 결과 생성
- 의미론적으로 중요한 에지를 더 명확히 감지

반면, Canny는 계산이 더 빠르고 추가 학습 없이 사용할 수 있다는 장점이 있습니다.

## 5. 실험 결과

이 코드를 실행하면 다음과 같은 결과를 얻을 수 있습니다:
- 원본 얼굴 이미지
- HED 모델로 추출한 에지 이미지
- Canny 알고리즘으로 추출한 에지 이미지
- 원본 이미지와 HED 에지의 오버레이 이미지

실험 결과, HED 모델은 얼굴의 주요 특징(눈, 코, 입, 얼굴 윤곽선 등)을 Canny보다 더 자연스럽고 일관되게 검출하는 것을 확인할 수 있습니다. 특히 얼굴의 미세한 특징까지 감지하며, 배경과 얼굴 경계를 더 명확하게 구분합니다.

## 6. 결론

이 프로젝트에서는 HED(Holistically-Nested Edge Detection) 모델을 사용하여 얼굴 이미지의 에지를 검출하는 방법을 구현했습니다. HED 모델은 다중 스케일 특징 추출과 깊은 감독(Deep Supervision) 기법을 통해 전통적인 에지 검출 방법보다 더 자연스럽고 의미론적인 에지를 검출할 수 있습니다.

코드는 OpenCV의 DNN 모듈을 활용하여 간결하게 구현되었으며, 사전 학습된 모델을 통해 추가적인 학습 없이도 높은 품질의 에지 검출 결과를 얻을 수 있습니다. 또한 기존의 Canny 에지 검출기와의 비교를 통해 딥러닝 기반 방식의 우수성을 확인할 수 있었습니다.

이 구현을 통해 컴퓨터 비전에서 딥러닝 기반 접근 방식이 전통적인 알고리즘보다 더 효과적일 수 있음을 배울 수 있었으며, 향후 다양한 이미지 처리 및 컴퓨터 비전 작업에 딥러닝 기술을 활용할 수 있는 기반을 마련했습니다.