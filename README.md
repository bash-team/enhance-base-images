# ✨ Enhance Base Images

Gemini API를 이용한 이미지 일괄 편집 CLI 도구

## 설치

### pip로 설치

```bash
pip install enhance-base-images
```

### 소스에서 설치

```bash
git clone https://github.com/jaeseokk/enhance-base-images.git
cd enhance-base-images
pip install -e .
```

## 빠른 시작

### 1. API 키 설정

```bash
# Keychain에 API 키 저장 (권장)
enhance-base-images config set-key

# 또는 환경 변수로 설정
export GOOGLE_API_KEY="your-api-key"
```

API 키는 [Google AI Studio](https://aistudio.google.com/apikey)에서 발급받을 수 있습니다.

### 2. 이미지 편집 실행

```bash
# 현재 폴더의 이미지를 편집
enhance-base-images run

# 특정 폴더 지정
enhance-base-images run -i ./input_images -o ./output_images

# 병렬 처리 수 조정
enhance-base-images run -w 5

# 확인 없이 바로 실행
enhance-base-images run -y
```

## 명령어

### `run` - 이미지 일괄 편집

```bash
enhance-base-images run [OPTIONS]
```

| 옵션 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `--input` | `-i` | 입력 이미지 폴더 | `.` (현재 폴더) |
| `--output` | `-o` | 출력 이미지 폴더 | `./edited` |
| `--model` | `-m` | 사용할 모델 | `gemini-3-pro-image-preview` |
| `--workers` | `-w` | 동시 처리 수 | `3` |
| `--prompt` | `-p` | 커스텀 프롬프트 | - |
| `--prompt-file` | - | 프롬프트 파일 경로 | - |
| `--yes` | `-y` | 확인 없이 실행 | `False` |
| `--debug` | `-d` | 디버그 모드 | `False` |

### `config` - API 키 관리

```bash
# API 키 설정
enhance-base-images config set-key

# 저장된 키 확인
enhance-base-images config show-key

# API 키 삭제
enhance-base-images config delete-key

# API 키 유효성 테스트
enhance-base-images config test
```

### `models` - 모델 목록

```bash
enhance-base-images models
```

| 모델 | 특징 |
|------|------|
| `gemini-3-pro-image-preview` | 높은 품질 (기본값) |
| `gemini-2.5-flash-image` | 빠른 처리 속도 |

### `info` - 이미지 정보

```bash
# 현재 폴더
enhance-base-images info

# 특정 폴더
enhance-base-images info ./images
```

## 커스텀 프롬프트 사용

### 명령줄에서 직접 입력

```bash
enhance-base-images run --prompt "Make the image brighter and more vibrant"
```

### 파일에서 읽기

```bash
# prompt.txt 파일 생성
echo "Enhance the colors and add a warm tone" > prompt.txt

# 프롬프트 파일 사용
enhance-base-images run --prompt-file prompt.txt
```

## API 키 우선순위

1. **환경 변수** (`GOOGLE_API_KEY`) - 최우선
2. **Keychain** - 환경 변수가 없을 때
3. **Interactive 입력** - 둘 다 없을 때 (저장 여부 선택 가능)

## 요구사항

- Python 3.10 이상
- Google AI API 키

## 라이선스

MIT License
