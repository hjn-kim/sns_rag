#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WhatsApp 채팅 CSV -> SNS 스키마 JSON + 임베딩용 청크 파일

WhatsApp 내보내기는 대화 경계가 없는 연속 로그다. 이것을 시간 간격으로 세션을
끊어 sns_sample.json 과 같은 구조로 바꾼다. 구조가 같으므로 gpu_translator.py 와
embedding.py 를 고치지 않고 그대로 쓸 수 있다.

    data/old/WhatsappChat.csv
      -> data/WhatsappChat.json      정규 구조 (번역 입력)
      -> data/Whatsapp_chat/         임베딩용 청크 (세션마다 파일 하나)

화자는 원본 이름을 그대로 쓴다. 익명화도 마스킹도 하지 않는다.

    "Oluwatobi Williams RESAGRATIA: Good morning Fam.."
    "+234 805 230 5080: Sounds great"

청크를 세션마다 파일 하나로 쪼개는 이유:
    embedding.py 는 폴더를 주면 파일 단위로 청킹한다. 세션 하나가 파일 하나면
    청킹이 세션 경계를 넘지 않는다. 한 파일에 다 넣으면 세션 A 의 끝과 세션 B 의
    시작이 한 청크에 섞인다.

번역된 JSON 에서 청크만 다시 만들려면 --data 에 .json 을 주면 된다:

    python src/whatsapp_to_json.py --data data/WhatsappChat_en.json \\
                                   --chunks data/Whatsapp_chat_en

사용 예:
    python src/whatsapp_to_json.py                      # 6시간 간격으로 세션 분할
    python src/whatsapp_to_json.py --gap 60             # 1시간 간격 (세션이 잘게 쪼개짐)
    python src/whatsapp_to_json.py --dry-run            # 파일을 쓰지 않고 통계만
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------

# 세션을 끊을 기본 간격(분). 6시간 이상 조용하면 다른 대화로 본다.
DEFAULT_GAP = 360

# 내용이 없는 메시지. 그대로 두면 "<Media omitted>" 같은 청크가 생긴다.
DROP_EXACT = {"<media omitted>", "null", "http", "https"}
DROP_CONTAINS = ("this message was deleted", "you deleted this message")

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------
# CSV 읽기
# --------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """(시각, 화자, 본문) 목록을 시간순으로 돌려준다."""
    with path.open(encoding="utf-8", errors="replace", newline="") as fp:
        rows = list(csv.DictReader(fp))

    if not rows:
        sys.exit(f"빈 CSV 입니다: {path}")
    for col in ("DateTime", "Name", "Content"):
        if col not in rows[0]:
            sys.exit(f"'{col}' 컬럼이 없습니다. 있는 컬럼: {list(rows[0])}")

    out: list[dict] = []
    unparsed = 0
    for r in rows:
        try:
            when = datetime.datetime.strptime(r["DateTime"].strip(), TIME_FORMAT)
        except ValueError:
            unparsed += 1
            continue
        out.append({
            "when": when,
            # CSV 의 Name/Content 앞뒤에 공백이 붙어 있다
            "name": r["Name"].strip(),
            "text": r["Content"].strip(),
        })
    if unparsed:
        print(f"  [!] DateTime 을 읽지 못한 {unparsed}행은 건너뛰었습니다.")

    out.sort(key=lambda m: m["when"])
    return out


def is_noise(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    if low in DROP_EXACT:
        return True
    return any(k in low for k in DROP_CONTAINS)


# --------------------------------------------------------------------------
# 세션 분할
# --------------------------------------------------------------------------

def split_sessions(messages: list[dict], gap_minutes: float) -> list[list[dict]]:
    """마지막 메시지로부터 gap 분 넘게 조용하면 새 세션으로 본다."""
    if not messages:
        return []
    sessions: list[list[dict]] = []
    current = [messages[0]]
    for prev, msg in zip(messages, messages[1:]):
        if (msg["when"] - prev["when"]).total_seconds() / 60 > gap_minutes:
            sessions.append(current)
            current = []
        current.append(msg)
    sessions.append(current)
    return sessions


# --------------------------------------------------------------------------
# SNS 스키마로 조립
# --------------------------------------------------------------------------

def build_document(sessions: list[list[dict]]) -> dict:
    """sns_sample.json 과 같은 {numberOfItems, data} 구조를 만든다."""
    items = []
    for si, session in enumerate(sessions):
        # 세션에 등장한 순서대로. 이름을 그대로 participantID 로 쓴다.
        participants: list[str] = []
        for m in session:
            if m["name"] not in participants:
                participants.append(m["name"])

        body = []
        turn = 0
        last_speaker = None
        for ui, m in enumerate(session, 1):
            if m["name"] != last_speaker:
                turn += 1
                last_speaker = m["name"]
            body.append({
                "utteranceID": f"U{ui}",
                "turnID": f"T{turn}",
                "participantID": m["name"],
                "date": m["when"].strftime("%Y-%m-%d"),
                "time": m["when"].strftime("%H:%M:%S"),
                "utterance": m["text"],
            })

        items.append({
            "header": {
                "dialogueInfo": {
                    "dialogueID": f"s{si:04d}",
                    "numberOfParticipants": len(participants),
                    "numberOfUtterances": len(body),
                    "numberOfTurns": turn,
                    "startedAt": session[0]["when"].strftime(TIME_FORMAT),
                    "endedAt": session[-1]["when"].strftime(TIME_FORMAT),
                },
                "participantsInfo": [{"participantID": p} for p in participants],
            },
            "body": body,
        })

    return {"numberOfItems": len(items), "data": items}


# --------------------------------------------------------------------------
# 임베딩용 청크 파일
# --------------------------------------------------------------------------

def session_text(item: dict) -> str:
    """세션 하나를 임베딩할 텍스트 한 덩어리로 만든다."""
    lines = []
    for u in item["body"]:
        text = u["utterance"].strip()
        if text:
            lines.append(f"{u['participantID']}: {text}")
    return "\n".join(lines)


def write_chunk_files(doc: dict, out_dir: Path) -> tuple[int, int]:
    """
    세션마다 JSON 하나를 쓴다. (파일 수, 빈 세션 수)

    embedding.py 는 최상위 "text" key 를 본문으로 잡으므로 그 이름을 쓴다.
    나머지 key 는 임베딩에 들어가지 않고 나중에 결과를 되짚을 때만 쓴다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = empty = 0
    for item in doc["data"]:
        text = session_text(item)
        if not text.strip():
            empty += 1
            continue
        info = item["header"]["dialogueInfo"]
        payload = {
            "text": text,
            "dialogueID": info["dialogueID"],
            "startedAt": info["startedAt"],
            "endedAt": info["endedAt"],
            "numberOfUtterances": info["numberOfUtterances"],
            "participants": [p["participantID"]
                             for p in item["header"]["participantsInfo"]],
        }
        with (out_dir / f"{info['dialogueID']}.json").open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        written += 1
    return written, empty


def clear_stale(out_dir: Path, keep: set[str]) -> int:
    """--gap 을 바꾸면 세션 수가 줄어든다. 지난 실행이 남긴 파일을 지운다."""
    if not out_dir.is_dir():
        return 0
    removed = 0
    for path in out_dir.glob("s*.json"):
        if path.name not in keep:
            path.unlink()
            removed += 1
    return removed


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------

def report(doc: dict) -> None:
    sizes = sorted(it["header"]["dialogueInfo"]["numberOfUtterances"] for it in doc["data"])
    chars = sorted(sum(len(u["utterance"]) for u in it["body"]) for it in doc["data"])
    speakers = sorted(it["header"]["dialogueInfo"]["numberOfParticipants"] for it in doc["data"])
    mid = len(sizes) // 2
    print(f"  세션 {len(sizes)}개 | 메시지 합계 {sum(sizes):,}")
    print(f"    세션당 메시지  중간값 {sizes[mid]}  최대 {sizes[-1]}")
    print(f"    세션당 글자    중간값 {chars[mid]:,}  최대 {chars[-1]:,}")
    print(f"    세션당 화자    중간값 {speakers[mid]}  최대 {speakers[-1]}")


def emit_chunks(doc: dict, chunks_dir: Path, dry_run: bool) -> None:
    if dry_run:
        n = sum(1 for it in doc["data"] if session_text(it).strip())
        print(f"  {chunks_dir.name + '/':<28} 세션 {n}개 (dry-run)")
        return
    keep = {f"{it['header']['dialogueInfo']['dialogueID']}.json" for it in doc["data"]}
    stale = clear_stale(chunks_dir, keep)
    written, empty = write_chunk_files(doc, chunks_dir)
    note = f", 빈 세션 {empty}개 건너뜀" if empty else ""
    note += f", 지난 실행 파일 {stale}개 삭제" if stale else ""
    print(f"  {chunks_dir.name + '/':<28} 파일 {written}개{note}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(
        description="WhatsApp CSV 를 SNS 스키마 JSON 과 임베딩용 청크 파일로 바꾼다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/old/WhatsappChat.csv", type=Path,
        help="원본 CSV. 이미 만든 구조 JSON(번역본 포함)을 주면 청크만 다시 만든다\n"
             "(기본: data/old/WhatsappChat.csv)",
    )
    parser.add_argument(
        "--out", default="data/WhatsappChat.json", type=Path,
        help="만들 구조 JSON 경로 (기본: data/WhatsappChat.json)",
    )
    parser.add_argument(
        "--chunks", default="data/Whatsapp_chat", type=Path,
        help="임베딩용 청크를 저장할 폴더 (기본: data/Whatsapp_chat)",
    )
    parser.add_argument(
        "--gap", type=float, default=DEFAULT_GAP,
        help=f"이만큼(분) 조용하면 새 세션으로 본다 (기본: {DEFAULT_GAP} = 6시간)\n"
             "작게 줄수록 세션이 잘게 쪼개진다",
    )
    parser.add_argument(
        "--keep-noise", action="store_true",
        help="<Media omitted>, 삭제된 메시지, URL 만 남은 메시지도 그대로 둔다",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="파일을 쓰지 않고 통계만 출력한다",
    )
    args = parser.parse_args()

    src: Path = args.data.resolve()
    if not src.is_file():
        sys.exit(f"파일을 찾을 수 없습니다: {src}")

    # ---- 이미 만든 구조 JSON 이면 청크만 다시 만든다 ----------------------
    if src.suffix.lower() == ".json":
        with src.open(encoding="utf-8") as fp:
            doc = json.load(fp)
        if not isinstance(doc, dict) or "data" not in doc:
            sys.exit("구조 JSON 이 아닙니다. {numberOfItems, data} 형태여야 합니다.")
        print(f"원본  : {src}")
        report(doc)
        emit_chunks(doc, args.chunks.resolve(), args.dry_run)
        return

    # ---- CSV -> 구조 JSON -------------------------------------------------
    print(f"원본  : {src}")
    print(f"세션  : {args.gap:.0f}분 이상 공백이면 분할")

    messages = load_csv(src)
    total = len(messages)
    if not args.keep_noise:
        messages = [m for m in messages if not is_noise(m["text"])]
    print(f"  메시지 {total:,}개 -> 정리 후 {len(messages):,}개")

    speakers = {m["name"] for m in messages}
    print(f"  화자 {len(speakers)}명 (원본 이름 그대로)")

    doc = build_document(split_sessions(messages, args.gap))
    report(doc)

    if args.dry_run:
        emit_chunks(doc, args.chunks.resolve(), True)
        print("\nDRY-RUN: 파일을 쓰지 않았습니다.")
        return

    dst: Path = args.out.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fp:
        json.dump(doc, fp, ensure_ascii=False, indent=2)
    print(f"  {dst.name:<28} 세션 {doc['numberOfItems']}개")

    emit_chunks(doc, args.chunks.resolve(), False)

    print("\n다음 단계:")
    print(f"  임베딩 python src/embedding.py --data {args.chunks}")


if __name__ == "__main__":
    main()
