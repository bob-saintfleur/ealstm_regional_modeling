import datetime
import pandas


def get_list_sub_date(period: tuple, seq_len: int):
    """
    Given a period with (start,end) as YYYYMMDD format, and a seq length, return all sub-period tuple as (start, end)
    dates according to seq-length
    Parameters
    ----------
        period: (yyyymmdd, yyyymmdd) evaluation period
        seq_len: sequence lengt for the lstm

    -------

    """
    ref_date = pd.date_range(start=pd.to_datetime(period[0], format="%Y%m%d"),
                             end=pd.to_datetime(period[1], format="%Y%m%d") + timedelta(days=7))
    lim_date = [((a + timedelta(days=-seq_len)).strftime("%d%m%Y"), a.strftime("%d%m%Y")) for a in ref_date if
                a.strftime("%m%d") != "0229"]
    return lim_date


def decrease_member_year(member: tuple, step: int = -1):
    """ Upgrade with step-year the given sub-period considered as a member"""
    member = (pd.to_datetime(member[0], format="%d%m%Y"), pd.to_datetime(member[1], format="%d%m%Y"))
    member = (f'{member[0].strftime("%d%m")}{int(member[0].strftime("%Y")) + step}',
              f'{member[1].strftime("%d%m")}{int(member[1].strftime("%Y")) + step}')
    member = (pd.to_datetime(member[0], format="%d%m%Y").strftime("%d%m%Y"),
              pd.to_datetime(member[1], format="%d%m%Y").strftime("%d%m%Y"))
    return member


def change_member_year(member: tuple, new_start_year: int = None):
    """ Use the new_start_year parameter to change the member's start year of the member '"""
    member = (pd.to_datetime(member[0], format="%d%m%Y"), pd.to_datetime(member[1], format="%d%m%Y"))
    if new_start_year is not None:
        step = int(new_start_year) - int(member[0].strftime("%Y"))
    else:
        step = 0
    member = (f'{member[0].strftime("%d%m")}{int(member[0].strftime("%Y")) + step}',
              f'{member[1].strftime("%d%m")}{int(member[1].strftime("%Y")) + step}')

    member = (pd.to_datetime(member[0], format="%d%m%Y").strftime("%d%m%Y"),
              pd.to_datetime(member[1], format="%d%m%Y").strftime("%d%m%Y"))
    return member
