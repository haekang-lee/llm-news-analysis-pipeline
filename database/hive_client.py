import jaydebeapi
import pandas as pd
import numpy as np
from omegaconf import DictConfig


def _get_connection(cfg: DictConfig):
    return jaydebeapi.connect(
        jclass_name="org.apache.hive.jdbc.HiveDriver",
        url=cfg.hive.url,
        driver_args=[cfg.hive.user, cfg.hive.passwd],
        jars=cfg.hive.jars
    )


def fetch_data(cfg: DictConfig, query: str) -> pd.DataFrame:
    conn = _get_connection(cfg)
    cursor = conn.cursor()
    df = pd.read_sql(query, conn)
    cursor.close()
    conn.close()

    df = df.replace('', np.nan)
    return df


def execute_ddl(cfg: DictConfig, ddl: str) -> None:
    """CREATE TABLE / DROP TABLE 등 DDL 실행"""
    conn = _get_connection(cfg)
    cursor = conn.cursor()
    try:
        cursor.execute(ddl)
        print("[완료] DDL 실행 성공")
    finally:
        cursor.close()
        conn.close()


def _escape(val) -> str:
    """값을 Hive INSERT용 SQL 리터럴로 변환"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "NULL"
    if isinstance(val, str):
        # Hive 문자열 이스케이프 (작은따옴표, 백슬래시, 개행문자 등 처리)
        escaped = val.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", " ")
        return f"'{escaped}'"
    return str(val)


def insert_dataframe(
    cfg: DictConfig,
    df: pd.DataFrame,
    table: str,
    batch_size: int = 1000,
    verbose: bool = True,
    partition_col: str = None,
) -> None:
    """
    DataFrame을 Hive 테이블에 배치 INSERT.
    - partition_col이 주어지면 동적 파티션 INSERT를 수행. (DataFrame의 마지막 컬럼이어야 함)
    """
    df = df.where(pd.notnull(df), None)  # NaN → None

    conn = _get_connection(cfg)
    cursor = conn.cursor()
    total = len(df)

    try:
        # 동적 파티션 설정 활성화
        if partition_col:
            cursor.execute("set hive.exec.dynamic.partition.mode=nonstrict")
            cursor.execute("set hive.exec.dynamic.partition=true")

        for start in range(0, total, batch_size):
            batch = df.iloc[start : start + batch_size]

            values_list = []
            for row in batch.itertuples(index=False, name=None):
                values = ", ".join(_escape(v) for v in row)
                values_list.append(f"({values})")

            if partition_col:
                sql = f"INSERT INTO TABLE {table} PARTITION ({partition_col}) VALUES {', '.join(values_list)}"
            else:
                sql = f"INSERT INTO {table} VALUES {', '.join(values_list)}"
                
            cursor.execute(sql)

            if verbose:
                end = min(start + batch_size, total)
                print(f"  [{end:>7,} / {total:,}] 삽입 완료")

        print(f"[완료] 총 {total:,}건 삽입 성공 → {table}")
    except Exception as e:
        print(f"[에러] INSERT 실패: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def insert_dataframe_with_id(
    cfg: DictConfig,
    df: pd.DataFrame,
    table: str,
    id_column: str = "news_mppg_id",
    batch_size: int = 1000,
    verbose: bool = True,
    partition_col: str = None,
) -> None:
    """
    DB의 현재 MAX(id_column)을 조회한 뒤,
    이어지는 순번을 DataFrame 첫 컬럼에 붙여서 배치 INSERT.
    """
    from queries.sql_queries import get_max_news_mppg_id

    max_id_df = fetch_data(cfg, get_max_news_mppg_id(table))
    max_id = int(max_id_df["max_id"].iloc[0] or 0)

    df = df.reset_index(drop=True)
    df.insert(0, id_column, range(max_id + 1, max_id + 1 + len(df)))

    if verbose:
        print(f"  현재 MAX({id_column}) = {max_id:,}  →  {max_id + 1:,} 부터 부여")

    insert_dataframe(cfg, df, table=table, batch_size=batch_size, verbose=verbose, partition_col=partition_col)