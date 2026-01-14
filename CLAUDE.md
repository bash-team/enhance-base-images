# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gemini API를 사용한 이미지 일괄 편집 CLI 도구. Typer + Rich 기반 CLI로, 병렬 처리를 통해 여러 이미지를 동시에 편집한다.

## Commands

```bash
# 개발 환경 설치
pip install -e ".[dev]"

# 빌드
python -m build

# 린트
ruff check enhance_base_images/
black --check enhance_base_images/

# CLI 실행 (개발 중)
python -m enhance_base_images.cli --help
```

## Architecture

단일 모듈 구조 (`enhance_base_images/cli.py`):

- **API 키 관리**: keyring을 통한 시스템 키체인 저장, 환경변수 폴백
- **프리셋/히스토리**: `~/.config/enhance-base-images/`에 JSON으로 저장
- **이미지 처리**: ThreadPoolExecutor로 병렬 처리, google-genai SDK 사용
- **CLI**: Typer 앱 + config/preset 서브커맨드 그룹

설정 파일 위치:
- `~/.config/enhance-base-images/presets.json` - 프롬프트 프리셋
- `~/.config/enhance-base-images/history.json` - 최근 사용 프롬프트 (5개)

## Release Process

```bash
# 1. 버전 업데이트 (__init__.py, pyproject.toml)
# 2. 빌드, 커밋, 태그, 푸시
python -m build && git add -A && git commit -m "chore: bump version to x.x.x" && git push origin main && git tag vx.x.x && git push origin vx.x.x

# 3. GitHub 릴리스 생성
gh release create vx.x.x --title "vx.x.x" --notes "RELEASE_NOTES" dist/enhance_base_images-x.x.x-py3-none-any.whl
```

### Release Note 템플릿

minor 버전 (기능 추가):
```
## 주요 변경사항

- 기능1
- 기능2

## 설치

\`\`\`bash
pipx install https://github.com/bash-team/enhance-base-images/releases/download/vx.x.x/enhance_base_images-x.x.x-py3-none-any.whl
\`\`\`

## 업데이트

\`\`\`bash
pipx uninstall enhance-base-images
pipx install https://github.com/bash-team/enhance-base-images/releases/download/vx.x.x/enhance_base_images-x.x.x-py3-none-any.whl
\`\`\`
```

patch 버전 (버그 수정/소규모 변경):
```
## 변경사항

- 변경 내용

## 설치

\`\`\`bash
pipx install https://github.com/bash-team/enhance-base-images/releases/download/vx.x.x/enhance_base_images-x.x.x-py3-none-any.whl
\`\`\`

## 업데이트

\`\`\`bash
pipx uninstall enhance-base-images
pipx install https://github.com/bash-team/enhance-base-images/releases/download/vx.x.x/enhance_base_images-x.x.x-py3-none-any.whl
\`\`\`
```
