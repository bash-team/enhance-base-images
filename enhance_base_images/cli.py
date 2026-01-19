#!/usr/bin/env python3
"""
Enhance Base Images - Gemini 이미지 편집 CLI
병렬 처리로 빠른 일괄 이미지 편집을 지원합니다.
"""
from __future__ import annotations

import os
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
import keyring
from InquirerPy import inquirer
from InquirerPy.separator import Separator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from PIL import Image
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import __version__

# ===== 앱 설정 =====
APP_NAME = "enhance-base-images"
KEYRING_SERVICE = "enhance-base-images"
KEYRING_USERNAME = "google_api_key"
CONFIG_DIR = Path.home() / ".config" / "enhance-base-images"
PRESETS_FILE = CONFIG_DIR / "presets.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
MAX_HISTORY = 5

# 초기 프리셋 (첫 실행 시 생성)
INITIAL_PRESET = {
    "face-enhance": {
        "description": "얼굴 선명도 향상",
        "prompt": """Adjust the subject's face to turn slightly toward the camera, approximately 8 degrees closer to a frontal angle, to improve facial clarity and recognition.
Allow the face to receive a subtle increase in light, gently lifting highlights without changing the overall lighting mood.
Enhance facial sharpness very slightly, improving definition and focus while keeping it consistent with the surrounding environment.
The face should remain natural and cohesive — no over-sharpening, no artificial contrast, and no separation from the rest of the image.

Keep everything else in the image exactly the same. Only modify the face as described above."""
    }
}

app = typer.Typer(
    name=APP_NAME,
    help="Gemini API를 이용한 이미지 일괄 편집 도구",
    add_completion=False,
)
console = Console()


# ===== API 키 관리 =====
def get_stored_api_key() -> Optional[str]:
    """Keychain에서 저장된 API 키 가져오기"""
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception:
        return None


def save_api_key(api_key: str) -> bool:
    """API 키를 Keychain에 저장"""
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)
        return True
    except Exception as e:
        console.print(f"[red]Keychain 저장 실패: {e}[/red]")
        return False


def delete_api_key() -> bool:
    """Keychain에서 API 키 삭제"""
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        return True
    except keyring.errors.PasswordDeleteError:
        return False
    except Exception as e:
        console.print(f"[red]삭제 실패: {e}[/red]")
        return False


def get_api_key(interactive: bool = True) -> str:
    """
    API 키 가져오기 (우선순위: 환경변수 → Keychain → 사용자 입력)
    """
    # 1. 환경 변수 확인
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key

    # 2. Keychain 확인
    api_key = get_stored_api_key()
    if api_key:
        return api_key

    # 3. Interactive 모드가 아니면 에러
    if not interactive:
        console.print("[red]API 키가 설정되지 않았습니다.[/red]")
        console.print("  [dim]'enhance-base-images config set-key' 명령으로 API 키를 설정하세요.[/dim]")
        raise typer.Exit(1)

    # 4. 사용자에게 입력 요청
    console.print()
    console.print("[yellow]API 키가 설정되지 않았습니다.[/yellow]")
    console.print("[dim]Google AI Studio에서 API 키를 발급받으세요: https://aistudio.google.com/apikey[/dim]")
    console.print()

    api_key = Prompt.ask(
        "[bold]Google API 키를 입력하세요[/bold]",
        password=True,
    )

    if not api_key or not api_key.strip():
        console.print("[red]API 키가 입력되지 않았습니다.[/red]")
        raise typer.Exit(1)

    api_key = api_key.strip()

    # Keychain에 저장할지 확인
    if typer.confirm("이 API 키를 Keychain에 저장하시겠습니까?", default=True):
        if save_api_key(api_key):
            console.print("[green]✓ API 키가 Keychain에 저장되었습니다.[/green]")
        else:
            console.print("[yellow]Keychain 저장에 실패했습니다. 이번 세션에서만 사용됩니다.[/yellow]")

    return api_key


def mask_api_key(api_key: str) -> str:
    """API 키 마스킹 (처음 4자리와 마지막 4자리만 표시)"""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


# ===== 프리셋 및 히스토리 관리 =====
def ensure_config_dir() -> None:
    """설정 디렉토리 생성"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_presets() -> dict:
    """프리셋 로드 (없으면 초기 프리셋 생성)"""
    ensure_config_dir()

    if not PRESETS_FILE.exists():
        save_presets(INITIAL_PRESET)
        return INITIAL_PRESET.copy()

    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_presets(presets: dict) -> bool:
    """프리셋 저장"""
    ensure_config_dir()
    try:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
        return True
    except IOError as e:
        console.print(f"[red]프리셋 저장 실패: {e}[/red]")
        return False


def load_history() -> list[dict]:
    """최근 사용 프롬프트 로드"""
    ensure_config_dir()

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("recent", [])
    except (json.JSONDecodeError, IOError):
        return []


def save_history(history: list[dict]) -> bool:
    """최근 사용 프롬프트 저장"""
    ensure_config_dir()
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"recent": history}, f, ensure_ascii=False, indent=2)
        return True
    except IOError:
        return False


def add_to_history(prompt: str) -> None:
    """프롬프트를 히스토리에 추가"""
    history = load_history()

    # 이미 존재하면 제거 (최신으로 올리기 위해)
    history = [h for h in history if h.get("prompt") != prompt]

    # 맨 앞에 추가
    history.insert(0, {
        "prompt": prompt,
        "used_at": datetime.now().isoformat()
    })

    # MAX_HISTORY 개수 유지
    history = history[:MAX_HISTORY]

    save_history(history)


def select_prompt_interactive() -> Optional[str]:
    """인터랙티브 프롬프트 선택 UI (방향키 사용)"""
    presets = load_presets()
    history = load_history()

    choices = []

    # 프리셋 추가
    if presets:
        choices.append(Separator("── 프리셋 ──"))
        for name, data in presets.items():
            desc = data.get("description", "")
            label = f"{name}  ({desc})" if desc else name
            choices.append({"name": label, "value": ("preset", data.get("prompt", ""))})

    # 최근 사용 추가
    if history:
        choices.append(Separator("── 최근 사용 ──"))
        for item in history:
            prompt_preview = item.get("prompt", "")[:50].replace("\n", " ")
            if len(item.get("prompt", "")) > 50:
                prompt_preview += "..."
            choices.append({"name": prompt_preview, "value": ("history", item.get("prompt", ""))})

    # 직접 입력 옵션
    choices.append(Separator("──────────"))
    choices.append({"name": "직접 입력", "value": ("direct", None)})

    if not presets and not history:
        # 프리셋도 히스토리도 없으면 바로 직접 입력
        return prompt_direct_input()

    console.print()
    result = inquirer.select(
        message="프롬프트 선택:",
        choices=choices,
        pointer="❯",
        qmark="",
        amark="",
    ).execute()

    if result is None:
        return None

    choice_type, prompt = result

    if choice_type == "direct":
        return prompt_direct_input()
    else:
        return prompt


def prompt_direct_input() -> Optional[str]:
    """직접 프롬프트 입력"""
    console.print()
    console.print("[dim]프롬프트를 입력하세요 (빈 줄 두 번으로 종료):[/dim]")

    lines = []
    empty_count = 0

    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break

    # 마지막 빈 줄들 제거
    while lines and lines[-1] == "":
        lines.pop()

    prompt = "\n".join(lines).strip()

    if not prompt:
        console.print("[red]프롬프트가 입력되지 않았습니다.[/red]")
        return None

    return prompt


# ===== 이미지 처리 =====
def get_image_files(input_dir: Path) -> list[Path]:
    """처리할 이미지 파일 목록 가져오기"""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []

    for ext in extensions:
        image_files.extend(input_dir.glob(ext))

    return sorted(image_files)


def edit_single_image(
    client,
    image_path: Path,
    output_path: Path,
    model: str,
    prompt: str,
) -> tuple[bool, str]:
    """단일 이미지 편집"""
    filename = image_path.name

    try:
        image_input = Image.open(image_path)

        response = client.models.generate_content(
            model=model,
            contents=[prompt, image_input],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data

                if isinstance(image_data, str):
                    image_data = base64.b64decode(image_data)

                edited_image = Image.open(BytesIO(image_data))
                edited_image.save(output_path)
                return True, filename

        return False, f"{filename}: 이미지 미반환"

    except Exception as e:
        return False, f"{filename}: {str(e)}"


# ===== CLI 명령어 =====
def version_callback(value: bool):
    if value:
        console.print(f"[cyan]Enhance Base Images[/cyan] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="버전 정보 표시",
    ),
):
    """Gemini API를 이용한 이미지 일괄 편집 도구"""
    pass


@app.command()
def run(
    input_dir: Path = typer.Option(
        Path("."),
        "--input", "-i",
        help="입력 이미지 폴더 경로",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("./edited"),
        "--output", "-o",
        help="출력 이미지 폴더 경로",
    ),
    model: str = typer.Option(
        "gemini-3-pro-image-preview",
        "--model", "-m",
        help="사용할 모델 (gemini-3-pro-image-preview 또는 gemini-2.5-flash-image)",
    ),
    workers: int = typer.Option(
        3,
        "--workers", "-w",
        help="동시 처리할 이미지 수",
        min=1,
        max=10,
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt", "-p",
        help="편집 프롬프트 (미지정시 인터랙티브 선택)",
    ),
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file",
        help="프롬프트가 담긴 텍스트 파일 경로",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    skip_confirm: bool = typer.Option(
        False,
        "--yes", "-y",
        help="확인 없이 바로 실행",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="디버그 모드 활성화",
    ),
):
    """
    이미지 일괄 편집을 실행합니다.

    예시:
        enhance-base-images run -i ./images -o ./output -w 5
        enhance-base-images run --model gemini-2.5-flash-image
        enhance-base-images run --prompt "Make it brighter"
    """
    # 헤더 출력
    console.print(Panel.fit(
        f"[bold cyan]✨ Enhance Base Images[/bold cyan] [dim]v{__version__}[/dim]\n[dim]Gemini 이미지 편집 CLI[/dim]",
        border_style="cyan",
    ))

    # API 키 가져오기 (interactive)
    api_key = get_api_key(interactive=True)
    client = genai.Client(api_key=api_key)

    # 출력 폴더 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프롬프트 결정
    if prompt_file:
        edit_prompt = prompt_file.read_text(encoding="utf-8").strip()
        console.print(f"[dim]프롬프트 파일 사용: {prompt_file}[/dim]")
    elif prompt:
        edit_prompt = prompt
    else:
        # 인터랙티브 선택
        edit_prompt = select_prompt_interactive()
        if not edit_prompt:
            raise typer.Exit(1)

    # 이미지 파일 목록
    image_files = get_image_files(input_dir)

    # 이미 처리된 파일 제외
    tasks = []
    skipped = 0
    for img_path in image_files:
        out_path = output_dir / f"{img_path.stem}_edited{img_path.suffix}"
        if out_path.exists():
            skipped += 1
        else:
            tasks.append((img_path, out_path))

    total = len(tasks)

    # 설정 테이블 출력
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("입력 폴더", str(input_dir.absolute()))
    table.add_row("출력 폴더", str(output_dir.absolute()))
    table.add_row("모델", model)
    table.add_row("병렬 처리", f"{workers}개 동시")
    table.add_row("처리 대상", f"{total}개 이미지")
    table.add_row("건너뜀", f"{skipped}개 (이미 존재)")

    console.print()
    console.print(table)
    console.print()

    if total == 0:
        console.print("[yellow]처리할 이미지가 없습니다.[/yellow]")
        raise typer.Exit(0)

    # 프롬프트 미리보기
    if debug:
        console.print(Panel(
            edit_prompt[:300] + ("..." if len(edit_prompt) > 300 else ""),
            title="편집 프롬프트",
            border_style="dim",
        ))

    # 사용자 확인
    if not skip_confirm:
        if not typer.confirm("계속 진행하시겠습니까?"):
            console.print("[yellow]취소되었습니다.[/yellow]")
            raise typer.Exit(0)

    # 처리 시작
    success_count = 0
    fail_count = 0
    failed_files = []
    start_time = time.time()

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("처리 중...", total=total)

        def process_image(args):
            img_path, out_path = args
            return edit_single_image(client, img_path, out_path, model, edit_prompt)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_image, task): task for task in tasks}

            for future in as_completed(futures):
                success, message = future.result()

                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_files.append(message)

                progress.update(task_id, advance=1)

                if debug and not success:
                    console.print(f"[red]  실패: {message}[/red]")

    elapsed = time.time() - start_time

    # 결과 요약
    console.print()
    result_table = Table(title="처리 결과", show_header=False, box=None)
    result_table.add_column(style="bold")
    result_table.add_column()
    result_table.add_row("✅ 성공", f"[green]{success_count}개[/green]")
    result_table.add_row("❌ 실패", f"[red]{fail_count}개[/red]" if fail_count > 0 else "0개")
    result_table.add_row("⏭️  건너뜀", f"{skipped}개")
    result_table.add_row("⏱️  소요 시간", f"{elapsed:.1f}초")
    if total > 0:
        result_table.add_row("📊 평균", f"{elapsed/total:.1f}초/장")

    console.print(result_table)

    if failed_files:
        console.print()
        console.print("[red]실패한 파일:[/red]")
        for f in failed_files:
            console.print(f"  [dim]• {f}[/dim]")

    # 히스토리에 추가
    if success_count > 0:
        add_to_history(edit_prompt)

    console.print()
    console.print(f"[green]출력 폴더: {output_dir.absolute()}[/green]")


@app.command()
def models():
    """사용 가능한 모델 목록을 표시합니다."""
    table = Table(title="사용 가능한 모델")
    table.add_column("모델명", style="cyan")
    table.add_column("특징")
    table.add_column("권장 용도")

    table.add_row(
        "gemini-3-pro-image-preview",
        "높은 품질",
        "고품질 결과물 필요시 (기본값)",
    )
    table.add_row(
        "gemini-2.5-flash-image",
        "빠른 처리 속도",
        "대량 이미지 처리, 테스트",
    )

    console.print(table)


@app.command()
def info(
    input_dir: Path = typer.Argument(
        Path("."),
        help="이미지 폴더 경로",
    ),
):
    """지정된 폴더의 이미지 정보를 표시합니다."""
    image_files = get_image_files(input_dir)

    if not image_files:
        console.print("[yellow]이미지 파일이 없습니다.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"이미지 목록 ({len(image_files)}개)")
    table.add_column("#", style="dim")
    table.add_column("파일명")
    table.add_column("크기", justify="right")
    table.add_column("해상도", justify="right")

    for i, img_path in enumerate(image_files[:20], 1):
        size_mb = img_path.stat().st_size / (1024 * 1024)
        try:
            with Image.open(img_path) as img:
                resolution = f"{img.width}x{img.height}"
        except Exception:
            resolution = "?"

        table.add_row(
            str(i),
            img_path.name[:40] + ("..." if len(img_path.name) > 40 else ""),
            f"{size_mb:.2f} MB",
            resolution,
        )

    if len(image_files) > 20:
        table.add_row("...", f"외 {len(image_files) - 20}개", "", "")

    console.print(table)


# ===== Config 서브커맨드 =====
config_app = typer.Typer(help="API 키 및 프리셋 설정 관리")
app.add_typer(config_app, name="config")

# 프리셋 서브커맨드
preset_app = typer.Typer(help="프롬프트 프리셋 관리")
config_app.add_typer(preset_app, name="preset")


@config_app.command("set-key")
def config_set_key():
    """API 키를 Keychain에 저장합니다."""
    console.print()
    console.print("[bold]Google API 키 설정[/bold]")
    console.print("[dim]Google AI Studio에서 API 키를 발급받으세요: https://aistudio.google.com/apikey[/dim]")
    console.print()

    # 기존 키 확인
    existing_key = get_stored_api_key()
    if existing_key:
        console.print(f"[yellow]기존 API 키가 있습니다: {mask_api_key(existing_key)}[/yellow]")
        if not typer.confirm("새 API 키로 덮어쓰시겠습니까?"):
            console.print("[dim]취소되었습니다.[/dim]")
            raise typer.Exit(0)

    api_key = Prompt.ask(
        "API 키를 입력하세요",
        password=True,
    )

    if not api_key or not api_key.strip():
        console.print("[red]API 키가 입력되지 않았습니다.[/red]")
        raise typer.Exit(1)

    api_key = api_key.strip()

    if save_api_key(api_key):
        console.print("[green]✓ API 키가 Keychain에 저장되었습니다.[/green]")
    else:
        console.print("[red]API 키 저장에 실패했습니다.[/red]")
        raise typer.Exit(1)


@config_app.command("show-key")
def config_show_key():
    """저장된 API 키 정보를 표시합니다."""
    console.print()

    # 환경 변수 확인
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        console.print(f"[bold]환경 변수:[/bold] {mask_api_key(env_key)} [green](활성)[/green]")
    else:
        console.print("[bold]환경 변수:[/bold] [dim]설정되지 않음[/dim]")

    # Keychain 확인
    stored_key = get_stored_api_key()
    if stored_key:
        status = "[dim](비활성 - 환경 변수 우선)[/dim]" if env_key else "[green](활성)[/green]"
        console.print(f"[bold]Keychain:[/bold]    {mask_api_key(stored_key)} {status}")
    else:
        console.print("[bold]Keychain:[/bold]    [dim]설정되지 않음[/dim]")

    if not env_key and not stored_key:
        console.print()
        console.print("[yellow]API 키가 설정되지 않았습니다.[/yellow]")
        console.print("[dim]'enhance-base-images config set-key' 명령으로 API 키를 설정하세요.[/dim]")


@config_app.command("delete-key")
def config_delete_key():
    """Keychain에서 API 키를 삭제합니다."""
    stored_key = get_stored_api_key()

    if not stored_key:
        console.print("[yellow]Keychain에 저장된 API 키가 없습니다.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]삭제할 API 키:[/bold] {mask_api_key(stored_key)}")

    if not typer.confirm("정말 삭제하시겠습니까?"):
        console.print("[dim]취소되었습니다.[/dim]")
        raise typer.Exit(0)

    if delete_api_key():
        console.print("[green]✓ API 키가 Keychain에서 삭제되었습니다.[/green]")
    else:
        console.print("[red]삭제에 실패했습니다.[/red]")
        raise typer.Exit(1)


@config_app.command("test")
def config_test():
    """API 키가 유효한지 테스트합니다."""
    console.print()
    console.print("[bold]API 키 테스트 중...[/bold]")

    try:
        api_key = get_api_key(interactive=False)
    except typer.Exit:
        return

    console.print(f"[dim]사용 중인 키: {mask_api_key(api_key)}[/dim]")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'API key is valid' in exactly those words.",
            config=types.GenerateImagesConfig(aspect_ratio="16:9")
        )
        console.print("[green]✓ API 키가 유효합니다![/green]")
        console.print(f"[dim]응답: {response.text[:50]}...[/dim]")
    except Exception as e:
        console.print(f"[red]✗ API 키 테스트 실패: {e}[/red]")
        raise typer.Exit(1)


# ===== Preset 서브커맨드 =====
@preset_app.command("list")
def preset_list():
    """저장된 프리셋 목록을 표시합니다."""
    presets = load_presets()

    if not presets:
        console.print("[yellow]저장된 프리셋이 없습니다.[/yellow]")
        console.print("[dim]'enhance-base-images config preset add' 명령으로 프리셋을 추가하세요.[/dim]")
        raise typer.Exit(0)

    table = Table(title="프롬프트 프리셋")
    table.add_column("이름", style="cyan bold")
    table.add_column("설명", style="dim")
    table.add_column("프롬프트 미리보기", style="dim italic")

    for name, data in presets.items():
        desc = data.get("description", "")
        prompt_preview = data.get("prompt", "")[:40].replace("\n", " ")
        if len(data.get("prompt", "")) > 40:
            prompt_preview += "..."
        table.add_row(name, desc, prompt_preview)

    console.print(table)
    console.print()
    console.print(f"[dim]설정 파일: {PRESETS_FILE}[/dim]")


@preset_app.command("add")
def preset_add():
    """새 프리셋을 추가합니다."""
    console.print()
    console.print("[bold]새 프리셋 추가[/bold]")
    console.print()

    # 이름 입력
    name = Prompt.ask("프리셋 이름")
    if not name or not name.strip():
        console.print("[red]이름이 입력되지 않았습니다.[/red]")
        raise typer.Exit(1)
    name = name.strip()

    # 기존 프리셋 확인
    presets = load_presets()
    if name in presets:
        if not typer.confirm(f"'{name}' 프리셋이 이미 존재합니다. 덮어쓰시겠습니까?"):
            console.print("[dim]취소되었습니다.[/dim]")
            raise typer.Exit(0)

    # 설명 입력
    description = Prompt.ask("설명 (선택사항)", default="")

    # 프롬프트 입력
    console.print()
    console.print("[dim]프롬프트를 입력하세요 (빈 줄 두 번으로 종료):[/dim]")

    lines = []
    empty_count = 0

    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            break

    # 마지막 빈 줄들 제거
    while lines and lines[-1] == "":
        lines.pop()

    prompt = "\n".join(lines).strip()

    if not prompt:
        console.print("[red]프롬프트가 입력되지 않았습니다.[/red]")
        raise typer.Exit(1)

    # 저장
    presets[name] = {
        "description": description,
        "prompt": prompt
    }

    if save_presets(presets):
        console.print(f"[green]✓ '{name}' 프리셋이 저장되었습니다.[/green]")
    else:
        raise typer.Exit(1)


@preset_app.command("delete")
def preset_delete(
    name: str = typer.Argument(..., help="삭제할 프리셋 이름"),
):
    """프리셋을 삭제합니다."""
    presets = load_presets()

    if name not in presets:
        console.print(f"[red]'{name}' 프리셋을 찾을 수 없습니다.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]삭제할 프리셋:[/bold] {name}")
    console.print(f"[dim]설명: {presets[name].get('description', '없음')}[/dim]")

    if not typer.confirm("정말 삭제하시겠습니까?"):
        console.print("[dim]취소되었습니다.[/dim]")
        raise typer.Exit(0)

    del presets[name]

    if save_presets(presets):
        console.print(f"[green]✓ '{name}' 프리셋이 삭제되었습니다.[/green]")
    else:
        raise typer.Exit(1)


@preset_app.command("show")
def preset_show(
    name: str = typer.Argument(..., help="확인할 프리셋 이름"),
):
    """프리셋의 전체 내용을 표시합니다."""
    presets = load_presets()

    if name not in presets:
        console.print(f"[red]'{name}' 프리셋을 찾을 수 없습니다.[/red]")
        raise typer.Exit(1)

    preset = presets[name]
    console.print()
    console.print(f"[bold cyan]{name}[/bold cyan]")
    if preset.get("description"):
        console.print(f"[dim]{preset['description']}[/dim]")
    console.print()
    console.print(Panel(preset.get("prompt", ""), title="프롬프트", border_style="dim"))


def cli():
    """CLI 엔트리포인트"""
    app()


if __name__ == "__main__":
    cli()
