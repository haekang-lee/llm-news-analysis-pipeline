import pandas as pd


def keep_latest_per_cust_no(df: pd.DataFrame, date_col: str = "lst_ases_dt") -> pd.DataFrame:
    """
    cust_no 기준 평가 접수일(date_col)이 가장 최신인 행만 남김.
    lst_ases_dt는 YYYYMMDD 문자열/숫자 모두 처리.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out[date_col] = pd.to_numeric(out[date_col], errors="coerce")
    out = out.dropna(subset=["cust_no", date_col])

    out = out.sort_values(["cust_no", date_col], ascending=[True, False])
    out = out.drop_duplicates(subset=["cust_no"], keep="first")

    out[date_col] = out[date_col].astype("Int64").astype(str)
    return out.reset_index(drop=True)

def rename_companies_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    사용할 컬럼들만 남기고 rename 하기
    """
    using_columns = ["cust_no", "cust_nm", "sglr_ases_modl_clas_cd_nm", "std_inds_clas_nm", "primy_prod_nm"]
    
    out = df.copy()
    out = out[using_columns]
    
    rename_columns = {
        'cust_no':'cust_no',
        'cust_nm':'cust_nm', 
        'sglr_ases_modl_clas_cd_nm':'sglr_ases_modl_clas_cd', 
        'std_inds_clas_nm':'std_inds_clas_nm',
        'primy_prod_nm':'primy_prod_nm'
    }
    
    out = out.rename(rename_columns, axis=1)
    return out