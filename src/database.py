"""Small persistence boundary. Feature builders operate on pandas, not DuckDB APIs."""
from contextlib import contextmanager
from pathlib import Path
import re
import duckdb
import pandas as pd
from .config import ROOT


def ident(value):
    if not re.fullmatch(r'[a-z_][a-z0-9_]*', value):
        raise ValueError('Unsafe SQL identifier: ' + value)
    return '"' + value + '"'

class Database:
    def __init__(self, path, read_only=False):
        if str(path) != ':memory:':
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path), read_only=read_only)
        self.con.execute("SET memory_limit='6GB'")
        self.con.execute('SET threads=4')
        self.con.execute('SET preserve_insertion_order=false')

    def initialize(self):
        self.con.execute((ROOT / 'sql/schema.sql').read_text())
        self.con.execute((ROOT / 'sql/pitch_tracking_join.sql').read_text())

    def close(self):
        self.con.close()

    def query(self, sql, params=None):
        return self.con.execute(sql, params or []).fetchdf()

    def scalar(self, sql, params=None):
        return self.con.execute(sql, params or []).fetchone()[0]

    @contextmanager
    def transaction(self):
        self.con.execute('BEGIN TRANSACTION')
        try:
            yield
            self.con.execute('COMMIT')
        except Exception:
            self.con.execute('ROLLBACK')
            raise

    def columns(self, table):
        return self.query('DESCRIBE ' + ident(table))['column_name'].tolist()

    def upsert(self, table, rows, keys):
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if df.empty:
            return
        df = df[[c for c in df.columns if c in self.columns(table)]].copy()
        if df.duplicated(keys).any():
            raise ValueError('Duplicate upsert keys for ' + table)
        self.con.register('_incoming', df)
        cols = ','.join(ident(c) for c in df.columns)
        conflict = ','.join(ident(c) for c in keys)
        # Updating an unchanged FK column forces a delete/insert in DuckDB and
        # fails for referenced parent rows. Exclude identical columns from UPDATE.
        join = ' AND '.join(f't.{ident(k)}=s.{ident(k)}' for k in keys)
        changed = []
        overlap = self.scalar(f'SELECT count(*) FROM {ident(table)} t JOIN _incoming s ON {join}')
        for col in (df.columns if overlap else []):
            if col not in keys and self.scalar(
                f'SELECT count(*) FROM {ident(table)} t JOIN _incoming s ON {join} '
                f'WHERE t.{ident(col)} IS DISTINCT FROM s.{ident(col)}'):
                changed.append(col)
        updates = ','.join(f'{ident(c)}=excluded.{ident(c)}' for c in changed)
        action = 'DO UPDATE SET ' + updates if updates else 'DO NOTHING'
        try:
            self.con.execute(f'INSERT INTO {ident(table)} ({cols}) SELECT {cols} FROM _incoming '
                             f'ON CONFLICT ({conflict}) {action}')
        finally:
            self.con.unregister('_incoming')

    def replace_derived(self, table, df, keys):
        """Retain stable key constraints; rebuild derived data atomically as one batch."""
        df = df.copy()
        self.con.register('_incoming', df)
        try:
            exists = self.scalar('SELECT count(*) FROM information_schema.tables WHERE table_name=?', [table])
            if not exists:
                types = self.query('DESCRIBE SELECT * FROM _incoming')
                definitions = [f'{ident(r.column_name)} {r.column_type}' for r in types.itertuples()]
                definitions.append('PRIMARY KEY (' + ','.join(ident(k) for k in keys) + ')')
                if 'game_id' in keys:
                    definitions.append('FOREIGN KEY (game_id) REFERENCES games(game_id)')
                if 'pitcher_id' in keys:
                    definitions.append('FOREIGN KEY (pitcher_id) REFERENCES pitchers(pitcher_id)')
                self.con.execute(f'CREATE TABLE {ident(table)} (' + ','.join(definitions) + ')')
            old = self.columns(table)
            # Additive migrations preserve the complete master feature contract.
            missing = set(old) - set(df.columns)
            if missing:
                raise RuntimeError(f'Refusing to discard existing derived columns: {sorted(missing)}')
            for r in self.query('DESCRIBE SELECT * FROM _incoming').itertuples():
                if r.column_name not in old:
                    self.con.execute(f'ALTER TABLE {ident(table)} ADD COLUMN {ident(r.column_name)} {r.column_type}')
            self.con.execute(f'DELETE FROM {ident(table)}')
            cols = ','.join(ident(c) for c in df.columns)
            self.con.execute(f'INSERT INTO {ident(table)} ({cols}) SELECT {cols} FROM _incoming')
        finally:
            self.con.unregister('_incoming')
