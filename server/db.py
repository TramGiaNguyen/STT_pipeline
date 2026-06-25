"""
Lớp lưu trữ kết quả STT vào PostgreSQL.

Thiết kế cố ý ĐƠN GIẢN & AN TOÀN:
  - Mỗi thao tác mở/đóng 1 connection riêng (không share connection giữa các thread —
    psycopg2 connection không thread-safe; STT chạy trong ThreadPoolExecutor).
  - Lưu là "best-effort": nếu DB chưa cấu hình hoặc lỗi thì CHỈ log cảnh báo, KHÔNG
    làm hỏng luồng nhận dạng STT. API truy vấn thì trả lỗi rõ ràng nếu DB không sẵn sàng.

Cấu hình qua biến môi trường:
  DATABASE_URL = postgresql://user:pass@host:5432/dbname
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

# Import psycopg2 mềm: nếu chưa cài (chạy local thiếu driver) thì tắt tính năng DB
# thay vì làm sập cả server.
try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
    _PSYCOPG_OK = True
except Exception as _e:  # pragma: no cover
    psycopg2 = None
    Json = None
    RealDictCursor = None
    _PSYCOPG_OK = False
    logger.warning("Không import được psycopg2 (%s) — tính năng lưu DB bị tắt.", _e)


def get_database_url():
    """Trả về DATABASE_URL nếu có cấu hình, ngược lại None."""
    return os.environ.get("DATABASE_URL") or None


def is_enabled():
    """DB chỉ bật khi vừa có driver psycopg2 vừa có DATABASE_URL."""
    return _PSYCOPG_OK and bool(get_database_url())


def _connect():
    """Mở 1 connection mới tới Postgres. Raise nếu không cấu hình/không kết nối được."""
    url = get_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL chưa được cấu hình")
    if not _PSYCOPG_OK:
        raise RuntimeError("Thiếu thư viện psycopg2")
    return psycopg2.connect(url)


_CREATE_SQL = """
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


def init_db(retries: int = 10, delay: float = 2.0):
    """
    Tạo bảng nếu chưa có. Gọi lúc server khởi động.
    Có retry vì Postgres (trong docker-compose) có thể chưa sẵn sàng ngay.
    Best-effort: thất bại thì log lỗi và bỏ qua (server vẫn chạy, chỉ là không lưu được).
    """
    if not is_enabled():
        logger.info("DB chưa bật (thiếu DATABASE_URL hoặc psycopg2) — bỏ qua init_db, không lưu lịch sử.")
        return False

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            conn = _connect()
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(_CREATE_SQL)
                logger.info("DB sẵn sàng: bảng 'transcriptions' đã tồn tại/được tạo.")
                return True
            finally:
                conn.close()
        except Exception as e:
            last_err = e
            logger.warning("init_db thử %d/%d chưa được (%s) — chờ %.1fs...", attempt, retries, e, delay)
            time.sleep(delay)
    logger.error("init_db thất bại sau %d lần thử: %s. Server vẫn chạy nhưng KHÔNG lưu được.", retries, last_err)
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
    """
    Lưu 1 bản ghi kết quả STT. Trả về id (int) nếu thành công, None nếu không lưu được.
    Best-effort: nuốt lỗi (chỉ log) để không làm hỏng phản hồi STT cho người dùng.
    """
    if not is_enabled():
        logger.info("Bỏ qua lưu DB (chưa bật). File=%s", ten_file)
        return None
    try:
        conn = _connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO transcriptions
                        (id_cuoc_hop, ho_va_ten, ten_file, language, diarize, diarize_mode,
                         num_segments, duration_sec, transcript, segments)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id;
                    """,
                    (
                        id_cuoc_hop, ho_va_ten, ten_file, language, bool(diarize), diarize_mode,
                        int(num_segments or 0), duration_sec, transcript,
                        Json(segments) if segments is not None else None,
                    ),
                )
                new_id = cur.fetchone()[0]
            logger.info("Đã lưu transcription id=%s (file=%s, %s segment).", new_id, ten_file, num_segments)
            return new_id
        finally:
            conn.close()
    except Exception as e:
        logger.error("Lưu DB thất bại (file=%s): %s", ten_file, e)
        return None


def _row_to_dict(row, include_full=False):
    """Chuẩn hoá 1 row (RealDictRow) thành dict JSON-friendly cho API."""
    d = dict(row)
    ca = d.get("created_at")
    if ca is not None and hasattr(ca, "isoformat"):
        d["created_at"] = ca.isoformat()
    if not include_full:
        # Bản danh sách: bỏ segments (nặng), rút gọn transcript thành preview
        d.pop("segments", None)
        t = d.get("transcript") or ""
        d["transcript_preview"] = (t[:200] + "…") if len(t) > 200 else t
        d.pop("transcript", None)
    return d


def list_transcriptions(limit: int = 50, offset: int = 0, id_cuoc_hop=None, ho_va_ten=None):
    """
    Liệt kê các bản ghi (metadata + preview), mới nhất trước. Lọc tuỳ chọn theo
    id_cuoc_hop / ho_va_ten. Raise nếu DB không sẵn sàng (để API trả 503).
    """
    if not is_enabled():
        raise RuntimeError("DB chưa cấu hình (thiếu DATABASE_URL)")
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    where, params = [], []
    if id_cuoc_hop:
        where.append("id_cuoc_hop = %s")
        params.append(id_cuoc_hop)
    if ho_va_ten:
        where.append("ho_va_ten ILIKE %s")
        params.append(f"%{ho_va_ten}%")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    conn = _connect()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM transcriptions {where_sql};", params)
            total = cur.fetchone()["n"]
            cur.execute(
                f"""
                SELECT id, id_cuoc_hop, ho_va_ten, ten_file, language, diarize, diarize_mode,
                       num_segments, duration_sec, transcript, created_at
                FROM transcriptions
                {where_sql}
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s;
                """,
                params + [limit, offset],
            )
            items = [_row_to_dict(r, include_full=False) for r in cur.fetchall()]
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    finally:
        conn.close()


def get_transcription(rec_id: int):
    """Lấy đầy đủ 1 bản ghi theo id (gồm transcript + segments). None nếu không có."""
    if not is_enabled():
        raise RuntimeError("DB chưa cấu hình (thiếu DATABASE_URL)")
    conn = _connect()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, id_cuoc_hop, ho_va_ten, ten_file, language, diarize, diarize_mode,
                       num_segments, duration_sec, transcript, segments, created_at
                FROM transcriptions WHERE id = %s;
                """,
                (int(rec_id),),
            )
            row = cur.fetchone()
        return _row_to_dict(row, include_full=True) if row else None
    finally:
        conn.close()
