"""
Lớp lưu trữ kết quả STT — hỗ trợ 2 backend, tự chọn theo DATABASE_URL:

  - PostgreSQL : DATABASE_URL=postgresql://user:pass@host:5432/db  (production / docker)
  - SQLite     : DATABASE_URL=sqlite:///duong_dan.db  HOẶC KHÔNG đặt DATABASE_URL
                 → dùng file local <project>/stt_local.db. KHỎI cần docker/Postgres,
                   tiện check nhanh khi dev.

Thiết kế giữ nguyên: mỗi thao tác mở/đóng 1 connection riêng; lưu là "best-effort"
(lỗi DB chỉ log, KHÔNG làm hỏng luồng STT). API truy vấn thì raise rõ nếu DB không sẵn sàng.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

import sqlite3  # stdlib — luôn có, dùng cho backend local

# Import psycopg2 mềm: thiếu cũng không sao nếu chỉ chạy SQLite local.
try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    _PSYCOPG_OK = True
except Exception as _e:  # pragma: no cover
    psycopg2 = None
    Json = None
    RealDictCursor = None
    _PSYCOPG_OK = False
    logger.warning("Không import được psycopg2 (%s) — chỉ dùng được SQLite local.", _e)


def get_database_url():
    """Trả về DATABASE_URL nếu có cấu hình, ngược lại None."""
    return os.environ.get("DATABASE_URL") or None


def _backend():
    """'postgres' nếu DATABASE_URL trỏ Postgres; còn lại (sqlite:/// hoặc trống) → 'sqlite'."""
    url = get_database_url()
    if url and url.startswith(("postgres://", "postgresql://")):
        return "postgres"
    return "sqlite"


def _sqlite_path():
    """Đường dẫn file SQLite: lấy từ sqlite:///... nếu có, mặc định <project>/stt_local.db."""
    url = get_database_url()
    if url and url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    return str(Path(__file__).resolve().parent.parent / "stt_local.db")


def is_enabled():
    """Postgres cần psycopg2 + URL; SQLite (stdlib) thì luôn sẵn sàng."""
    if _backend() == "postgres":
        return _PSYCOPG_OK and bool(get_database_url())
    return True


def _connect():
    """Mở 1 connection mới theo backend đang chọn."""
    if _backend() == "postgres":
        if not _PSYCOPG_OK:
            raise RuntimeError("Thiếu thư viện psycopg2")
        return psycopg2.connect(get_database_url())
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    return conn


def _q(sql: str) -> str:
    """Đổi placeholder %s -> ? cho SQLite (Postgres giữ nguyên)."""
    return sql if _backend() == "postgres" else sql.replace("%s", "?")


def _dict_cursor(conn):
    """Cursor trả về row dạng mapping cho cả 2 backend."""
    if _backend() == "postgres":
        return conn.cursor(cursor_factory=RealDictCursor)
    return conn.cursor()  # conn.row_factory = sqlite3.Row đã set


def _as_dict(row) -> dict:
    """RealDictRow (pg) hoặc sqlite3.Row → dict thường."""
    if isinstance(row, dict):
        return dict(row)
    return {k: row[k] for k in row.keys()}


_CREATE_SQL_PG = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id            SERIAL PRIMARY KEY,
    id_cuoc_hop   TEXT,
    ho_va_ten     TEXT,
    ten_file      TEXT,
    language      TEXT,
    diarize       BOOLEAN     NOT NULL DEFAULT FALSE,
    diarize_mode  TEXT,
    num_segments  INTEGER     NOT NULL DEFAULT 0,
    duration_sec  DOUBLE PRECISION,
    transcript    TEXT,
    segments      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_transcriptions_id_cuoc_hop ON transcriptions (id_cuoc_hop);
CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at  ON transcriptions (created_at DESC);
"""

_CREATE_SQL_SQLITE = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    id_cuoc_hop   TEXT,
    ho_va_ten     TEXT,
    ten_file      TEXT,
    language      TEXT,
    diarize       INTEGER NOT NULL DEFAULT 0,
    diarize_mode  TEXT,
    num_segments  INTEGER NOT NULL DEFAULT 0,
    duration_sec  REAL,
    transcript    TEXT,
    segments      TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_transcriptions_id_cuoc_hop ON transcriptions (id_cuoc_hop);
CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at  ON transcriptions (created_at);
"""


def init_db(retries: int = 10, delay: float = 2.0):
    """Tạo bảng nếu chưa có. Gọi lúc server khởi động."""
    if not is_enabled():
        logger.info("DB chưa bật — bỏ qua init_db, không lưu lịch sử.")
        return False

    # SQLite: file local, không cần retry.
    if _backend() == "sqlite":
        try:
            conn = _connect()
            try:
                conn.executescript(_CREATE_SQL_SQLITE)
                conn.commit()
                logger.info("DB sẵn sàng (SQLite local): %s", _sqlite_path())
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error("init_db (SQLite) thất bại: %s", e)
            return False

    # Postgres: có retry vì container có thể chưa sẵn sàng ngay.
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = _connect()
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(_CREATE_SQL_PG)
                logger.info("DB sẵn sàng (Postgres): bảng 'transcriptions' OK.")
                return True
            finally:
                conn.close()
        except Exception as e:
            last_err = e
            logger.warning("init_db thử %d/%d chưa được (%s) — chờ %.1fs...", attempt, retries, e, delay)
            time.sleep(delay)
    logger.error("init_db (Postgres) thất bại sau %d lần: %s. Server vẫn chạy nhưng KHÔNG lưu được.", retries, last_err)
    return False


def save_transcription(
    *,
    id_cuoc_hop=None,
    ho_va_ten=None,
    ten_file=None,
    language=None,
    diarize=False,
    diarize_mode=None,
    num_segments=0,
    duration_sec=None,
    transcript=None,
    segments=None,
):
    """Lưu 1 bản ghi kết quả STT. Trả về id (int) nếu thành công, None nếu không lưu được."""
    if not is_enabled():
        logger.info("Bỏ qua lưu DB (chưa bật). File=%s", ten_file)
        return None

    b = _backend()
    # segments: pg dùng JSONB qua Json(), sqlite lưu chuỗi JSON.
    if segments is None:
        seg_param = None
    elif b == "postgres":
        seg_param = Json(segments)
    else:
        seg_param = json.dumps(segments, ensure_ascii=False)

    cols = ("(id_cuoc_hop, ho_va_ten, ten_file, language, diarize, diarize_mode, "
            "num_segments, duration_sec, transcript, segments)")
    vals = (
        id_cuoc_hop, ho_va_ten, ten_file, language, int(bool(diarize)), diarize_mode,
        int(num_segments or 0), duration_sec, transcript, seg_param,
    )

    try:
        conn = _connect()
        try:
            with conn:
                cur = conn.cursor()
                if b == "postgres":
                    cur.execute(
                        f"INSERT INTO transcriptions {cols} "
                        f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;",
                        vals,
                    )
                    new_id = cur.fetchone()[0]
                else:
                    cur.execute(
                        _q(f"INSERT INTO transcriptions {cols} "
                           f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"),
                        vals,
                    )
                    new_id = cur.lastrowid
            logger.info("Đã lưu transcription id=%s (file=%s, %s segment, backend=%s).",
                        new_id, ten_file, num_segments, b)
            return new_id
        finally:
            conn.close()
    except Exception as e:
        logger.error("Lưu DB thất bại (file=%s): %s", ten_file, e)
        return None


def _row_to_dict(row, include_full=False):
    """Chuẩn hoá 1 row thành dict JSON-friendly (xử lý cả pg và sqlite)."""
    d = _as_dict(row)

    # diarize về bool (sqlite lưu 0/1)
    if "diarize" in d and d["diarize"] is not None:
        d["diarize"] = bool(d["diarize"])

    # segments: sqlite lưu chuỗi JSON → parse lại
    seg = d.get("segments")
    if isinstance(seg, str):
        try:
            d["segments"] = json.loads(seg)
        except Exception:
            d["segments"] = None

    # created_at: pg là datetime → isoformat; sqlite đã là chuỗi
    ca = d.get("created_at")
    if ca is not None and hasattr(ca, "isoformat"):
        d["created_at"] = ca.isoformat()

    if not include_full:
        d.pop("segments", None)
        t = d.get("transcript") or ""
        d["transcript_preview"] = (t[:200] + "…") if len(t) > 200 else t
        d.pop("transcript", None)
    return d


def list_transcriptions(limit: int = 50, offset: int = 0, id_cuoc_hop=None, ho_va_ten=None):
    """Liệt kê các bản ghi (metadata + preview), mới nhất trước. Raise nếu DB không sẵn sàng."""
    if not is_enabled():
        raise RuntimeError("DB chưa cấu hình")
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    like_op = "ILIKE" if _backend() == "postgres" else "LIKE"

    where, params = [], []
    if id_cuoc_hop:
        where.append("id_cuoc_hop = %s")
        params.append(id_cuoc_hop)
    if ho_va_ten:
        where.append(f"ho_va_ten {like_op} %s")
        params.append(f"%{ho_va_ten}%")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = _connect()
    try:
        with conn:
            cur = _dict_cursor(conn)
            cur.execute(_q(f"SELECT COUNT(*) AS n FROM transcriptions {where_sql};"), params)
            total = _as_dict(cur.fetchone())["n"]
            cur.execute(
                _q(f"""
                SELECT id, id_cuoc_hop, ho_va_ten, ten_file, language, diarize, diarize_mode,
                       num_segments, duration_sec, transcript, created_at
                FROM transcriptions
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s;
                """),
                params + [limit, offset],
            )
            items = [_row_to_dict(r, include_full=False) for r in cur.fetchall()]
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    finally:
        conn.close()


def get_meeting_utterances(id_cuoc_hop):
    """
    Lấy TẤT CẢ lượt nói của 1 cuộc họp (theo id_cuoc_hop), sắp theo thời gian TĂNG dần.
    Dùng để ghép thành transcript hoàn chỉnh. Raise nếu DB không sẵn sàng.
    """
    if not is_enabled():
        raise RuntimeError("DB chưa cấu hình")
    conn = _connect()
    try:
        with conn:
            cur = _dict_cursor(conn)
            cur.execute(
                _q("""
                SELECT id, id_cuoc_hop, ho_va_ten, ten_file, language,
                       num_segments, duration_sec, transcript, segments, created_at
                FROM transcriptions
                WHERE id_cuoc_hop = %s
                ORDER BY created_at ASC, id ASC;
                """),
                (id_cuoc_hop,),
            )
            rows = cur.fetchall()
        return [_row_to_dict(r, include_full=True) for r in rows]
    finally:
        conn.close()


def list_meetings():
    """
    Liệt kê TẤT CẢ cuộc họp đang có (gom theo id_cuoc_hop): số lượt nói, danh sách người
    nói, tổng segment, thời điểm bắt đầu/kết thúc. Mới nhất trước. Dùng khi chưa biết ID.
    """
    if not is_enabled():
        raise RuntimeError("DB chưa cấu hình")
    conn = _connect()
    try:
        with conn:
            cur = _dict_cursor(conn)
            cur.execute("""
                SELECT id_cuoc_hop, ho_va_ten, num_segments, created_at
                FROM transcriptions
                WHERE id_cuoc_hop IS NOT NULL AND id_cuoc_hop <> ''
                ORDER BY created_at ASC, id ASC;
            """)
            rows = [_as_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    meetings = {}
    for r in rows:
        mid = r["id_cuoc_hop"]
        m = meetings.setdefault(mid, {
            "id_cuoc_hop": mid, "so_luot_noi": 0, "nguoi_noi": [],
            "tong_segments": 0, "bat_dau": None, "ket_thuc": None,
        })
        m["so_luot_noi"] += 1
        m["tong_segments"] += int(r.get("num_segments") or 0)
        name = (r.get("ho_va_ten") or "").strip()
        if name and name not in m["nguoi_noi"]:
            m["nguoi_noi"].append(name)
        ca = r.get("created_at")
        ca = ca.isoformat() if hasattr(ca, "isoformat") else ca
        if m["bat_dau"] is None:
            m["bat_dau"] = ca
        m["ket_thuc"] = ca

    return sorted(meetings.values(), key=lambda x: x["ket_thuc"] or "", reverse=True)


def get_transcription(rec_id: int):
    """Lấy đầy đủ 1 bản ghi theo id (gồm transcript + segments). None nếu không có."""
    if not is_enabled():
        raise RuntimeError("DB chưa cấu hình")
    conn = _connect()
    try:
        with conn:
            cur = _dict_cursor(conn)
            cur.execute(
                _q("""
                SELECT id, id_cuoc_hop, ho_va_ten, ten_file, language, diarize, diarize_mode,
                       num_segments, duration_sec, transcript, segments, created_at
                FROM transcriptions WHERE id = %s;
                """),
                (int(rec_id),),
            )
            row = cur.fetchone()
        return _row_to_dict(row, include_full=True) if row else None
    finally:
        conn.close()
