"""
Query-result captures.

A capture runs a single read query on its own connection and streams the
rows into a temporary SQLite database on disk. The stored result keeps the
column order (duplicate column names included) and the SQLite value types
of the original query, and is decoupled from later edits of the source
tables. Only a capture that ran to completion carries the 'complete'
marker in its store, so a partial or failed result can never be browsed
or exported as if it were complete.

Nothing here holds the full result in memory: capturing, pagination and
export all move rows in chunks.
"""

import base64
import csv
import json
import logging
import os
import secrets
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime
from io import StringIO

log = logging.getLogger(__name__)

CAPTURE_CHUNK = 1000  # Rows per fetch/insert batch while capturing.
EXPORT_CHUNK = 1000   # Rows per fetch batch while streaming an export.


class CaptureIncomplete(Exception):
    """The store holds no complete result (running, failed or destroyed)."""


class CaptureCancelled(Exception):
    pass


class CaptureError(Exception):
    pass


class CaptureStore(object):
    """
    On-disk temporary result of one capture: a SQLite database holding the
    rows positionally ("c0".."cn", no type affinity, so values keep their
    SQLite types) plus a _meta table with the original column names and
    row count. The 'complete' marker is written last, in its own
    transaction, and is what makes the result readable and exportable.
    """
    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix='sqlite-web-capture-')
        self.path = os.path.join(self.tmpdir, 'result.db')
        self._writer = None
        self._insert_sql = None

    #
    # Writer side. Only ever called from the task thread.
    #

    def begin(self, columns, sql):
        self._writer = sqlite3.connect(self.path, isolation_level=None)
        defs = ', '.join('"c%d"' % i for i in range(len(columns)))
        self._writer.execute('BEGIN')
        self._writer.execute(
            'CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)')
        self._writer.execute('CREATE TABLE _rows (%s)' % defs)
        self._writer.execute('INSERT INTO _meta VALUES (?, ?)',
                             ('columns', json.dumps(columns)))
        self._writer.execute('INSERT INTO _meta VALUES (?, ?)', ('sql', sql))
        self._writer.execute('COMMIT')
        self._insert_sql = 'INSERT INTO _rows VALUES (%s)' % (
            ', '.join(['?'] * len(columns)))

    def write_rows(self, rows):
        self._writer.execute('BEGIN')
        self._writer.executemany(self._insert_sql, rows)
        self._writer.execute('COMMIT')

    def finish(self, row_count):
        self._writer.execute('BEGIN')
        self._writer.execute('INSERT INTO _meta VALUES (?, ?)',
                             ('row_count', str(row_count)))
        self._writer.execute('INSERT INTO _meta VALUES (?, ?)',
                             ('complete', '1'))
        self._writer.execute('COMMIT')

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    #
    # Reader side. Called from request threads once the task is done.
    #

    def open_reader(self):
        # Open read-only so a reader can never recreate a destroyed store.
        conn = None
        try:
            conn = sqlite3.connect('file:%s?mode=ro' % self.path, uri=True)
            row = conn.execute(
                "SELECT value FROM _meta WHERE key = 'complete'").fetchone()
        except sqlite3.Error:
            if conn is not None:
                conn.close()
            raise CaptureIncomplete()
        if row is None:
            conn.close()
            raise CaptureIncomplete()
        return conn

    @staticmethod
    def read_columns(conn):
        row = conn.execute(
            "SELECT value FROM _meta WHERE key = 'columns'").fetchone()
        return json.loads(row[0])

    def read_page(self, page, page_size, ordering=None):
        conn = self.open_reader()
        try:
            columns = self.read_columns(conn)
            sql = 'SELECT * FROM _rows'
            if ordering:
                idx = abs(ordering)
                if 1 <= idx <= len(columns):
                    sql += ' ORDER BY %d %s' % (
                        idx, 'DESC' if ordering < 0 else 'ASC')
            sql += ' LIMIT %d OFFSET %d' % (
                page_size + 1, (max(page, 1) - 1) * page_size)
            rows = conn.execute(sql).fetchall()
            return columns, rows[:page_size], len(rows) > page_size
        finally:
            conn.close()

    def destroy(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class ExportStreamer(object):
    """
    Streams a completed store as CSV or JSON without loading it into
    memory. Owns its reader connection; close() is idempotent and safe to
    call even when iteration never started (HEAD requests, or a client
    that disconnects mid-stream).
    """
    def __init__(self, store, fmt, base64_bytes=False):
        self._conn = store.open_reader()  # Raises CaptureIncomplete.
        self._columns = store.read_columns(self._conn)
        self._fmt = fmt
        self._base64_bytes = base64_bytes

    def _value(self, value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value)
            if self._base64_bytes:
                return base64.urlsafe_b64encode(value).decode('utf8')
            return value.hex()
        return value

    def __iter__(self):
        try:
            if self._fmt == 'json':
                yield from self._iter_json()
            else:
                yield from self._iter_csv()
        finally:
            self.close()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _rows(self):
        cursor = self._conn.execute('SELECT * FROM _rows')
        while True:
            rows = cursor.fetchmany(EXPORT_CHUNK)
            if not rows:
                break
            for row in rows:
                yield row

    def _iter_csv(self):
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(self._columns)
        yield buf.getvalue()
        for row in self._rows():
            buf.seek(0)
            buf.truncate(0)
            writer.writerow([self._value(value) for value in row])
            yield buf.getvalue()

    def _iter_json(self):
        # An envelope, not a list of objects: column order and duplicate
        # column names survive, and an empty result still has its header.
        yield '{"columns": %s, "rows": [' % json.dumps(self._columns)
        first = True
        for row in self._rows():
            line = json.dumps([self._value(value) for value in row])
            yield ('\n' if first else ',\n') + line
            first = False
        yield ']}' if first else '\n]}'


class CaptureTask(object):
    """
    Runs one read query in the background and keeps the captured result.

    The task owns everything it uses: a dedicated connection to the
    database it was started on (built by the caller with the same settings
    as the app's own connections) and the temporary CaptureStore. Later
    dataset switches or unloads cannot redirect it, and close() releases
    both the connection and the stored result.
    """
    def __init__(self, dataset_key, dataset_name, sql, database,
                 startup_hook=None):
        self.id = secrets.token_urlsafe(8)
        self.dataset_key = dataset_key
        self.dataset_name = dataset_name
        self.sql = sql
        self.status = 'running'
        self.error = None
        self.row_count = 0
        self.columns = []
        self.created = datetime.now()
        self.store = CaptureStore()
        self._database = database  # Unconnected peewee database.
        self._startup_hook = startup_hook
        self._cancel = threading.Event()
        self._conn = None  # Raw connection, shared for interrupt().
        self._thread = threading.Thread(
            target=self._run, name='capture-%s' % self.id)
        self._thread.daemon = True

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancel.set()
        conn = self._conn
        if conn is not None:
            try:
                conn.interrupt()
            except Exception:
                pass

    def close(self):
        # Cancel, give the task thread a moment to unwind, then drop the
        # stored result. Safe to call more than once.
        self.cancel()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self.store.destroy()

    def _run(self):
        try:
            db = self._database
            db.connect()
            if self._startup_hook:
                self._startup_hook(db)
            self._conn = db.connection()
            cursor = db.execute_sql(self.sql.rstrip('; \t\r\n'))
            try:
                if cursor.description is None:
                    raise CaptureError('Statement did not return rows.')
                self.columns = [d[0] for d in cursor.description]
                self.store.begin(self.columns, self.sql)
                while True:
                    if self._cancel.is_set():
                        raise CaptureCancelled()
                    rows = cursor.fetchmany(CAPTURE_CHUNK)
                    if not rows:
                        break
                    self.store.write_rows(rows)
                    self.row_count += len(rows)
                self.store.finish(self.row_count)
            finally:
                cursor.close()
            self.status = 'done'
        except CaptureCancelled:
            self.status = 'cancelled'
        except Exception as exc:
            if self._cancel.is_set():
                self.status = 'cancelled'
            else:
                self.status = 'error'
                self.error = str(exc)
                log.exception('Capture %s failed.', self.id)
        finally:
            self._conn = None
            self.store.close_writer()
            try:
                self._database.close()
            except Exception:
                pass
            if self.status != 'done':
                # A partial or failed result is not a result: drop it.
                self.store.destroy()


#
# Registry of live captures, keyed by task id.
#

captures = {}
captures_lock = threading.Lock()


def register_task(task):
    with captures_lock:
        while task.id in captures:
            task.id = secrets.token_urlsafe(8)
        captures[task.id] = task
    return task


def get_task(task_id):
    with captures_lock:
        return captures.get(task_id)


def remove_task(task_id):
    with captures_lock:
        return captures.pop(task_id, None)


def tasks_for_dataset(dataset_key):
    with captures_lock:
        tasks = [task for task in captures.values()
                 if task.dataset_key == dataset_key]
    return sorted(tasks, key=lambda task: task.created, reverse=True)


def close_tasks_for_dataset(dataset_key):
    with captures_lock:
        tasks = [captures.pop(key) for key, task in list(captures.items())
                 if task.dataset_key == dataset_key]
    for task in tasks:
        task.close()


def close_all_tasks():
    with captures_lock:
        tasks = list(captures.values())
        captures.clear()
    for task in tasks:
        task.close()
