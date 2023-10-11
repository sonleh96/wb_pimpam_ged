import datetime
import dateutil
from dateutil.parser import parse
from dateutil.rrule import rrule, MONTHLY, WEEKLY, DAILY
import calendar
################################################################################
### Datetime string manipulation
################################################################################
#
def get_date_from_YW(input_YW, input_format="%YW%W", output_format="%Y%m%d"):
    """
    Example: get_date_from_YW('2016W10')
    """
    output=[]
    for day_in_week in range(7):
        t0_dt=datetime.datetime.strptime(input_YW + '-%d'%day_in_week, "%s-%%w"%input_format)
        output.append(t0_dt.strftime(output_format))
    output=sorted(output)
    return output
#
def get_prev_YW_from_YW(input_YW, week_delta_list, input_format="%YW%W", output_format="%YW%W"):
    """
    Example: get_prev_YW_from_YW('2016W10', [-2,-1])
    """
    output=[((datetime.datetime.strptime(input_YW+'-0', "%s-%%w"%input_format) + datetime.timedelta(days=(i*7)))
        .strftime(output_format))
        for i in week_delta_list]
    return output
#
def get_last_day_of_month_from_Ym(input_Ym, delta_month=0, input_format="%Y%m", output_format="%Y%m%d"):
    """
    Example: get_last_day_of_month_from_Ym('20161001', -1)
    """
    date01_nextmonth=(datetime.datetime.strptime(str(input_Ym)+'-28', input_format+"-%d")
        + dateutil.relativedelta.relativedelta(months=delta_month)
        + datetime.timedelta(days=4) )
    date02_lastdayofmonth=date01_nextmonth - datetime.timedelta(days=date01_nextmonth.day)
    return date02_lastdayofmonth.strftime(output_format)
#
def get_last_week_of_month_from_Ym(input_Ym, delta_month=0, input_format="%Y%m", output_format="%YW%W"):
    """
    Example: get_last_week_of_month_from_Ym('201610', -1)
    """
    date01=(datetime.datetime.strptime(str(input_Ym), input_format)
        + dateutil.relativedelta.relativedelta(months=delta_month+1)
        - dateutil.relativedelta.relativedelta(days=7) )
    return date01.strftime(output_format)
#
def get_prev_nmonths(upto_month, nmonth_lookback=2):
    """
    Example: get_prev_nmonths('201610', 2)
    """
    ym = []
    for i in range(-nmonth_lookback, 0, 1):
        ym.append(get_last_day_of_month_from_Ym(upto_month, delta_month=i+1)[0:6])
    return ym
#
def get_Ym_from_Ymd(input_Ymd, input_format="%Y%m%d",output_format="%Y%m"):
    """
    Example: get_Ym_from_Ymd("20160303")
    """
    return datetime.datetime.strptime(input_Ymd,input_format).strftime(output_format)

#
def get_Ymd_from_Ym(input_Ym,lookback,input_format="%Y%m",output_format="%Y%m%d"):
    """
    Example: get_ymd_from_Ym("201603",3)
    """
    date_list = []
    month_range = get_prev_nmonths(input_Ym, nmonth_lookback=lookback)
    for month in month_range:
        last_day = get_last_day_of_month_from_Ym(month,0,input_format,output_format)
        first_day = datetime.datetime.strptime(month+'01',input_format+"%d").strftime(output_format)
        dates_of_month = get_daterange(first_day,last_day)
        date_list += dates_of_month
    return date_list

#date generator
def get_daterange(start_date_str,end_date_str,input_format="%Y%m%d",output_format="%Y%m%d"):
    """
    Example: get_daterange("20160303","20160320")
    """
    end_date = datetime.datetime.strptime(end_date_str,input_format)
    start_date = datetime.datetime.strptime(start_date_str,input_format)
    #output = [(start_date+datetime.timedelta(n)).strftime(output_format) for n in range((end_date-start_date+datetime.timedelta(1)).days)]
    output = [d.strftime(output_format) for d in rrule(DAILY,dtstart=start_date,until=end_date)]
    return output

#date generator
def get_weekly_daterange(start_date_str,end_date_str,input_format="%Y%m%d",output_format="%Y%m%d"):
    """
    Example: get_daterange("20160303","20160320")
    """
    end_date = datetime.datetime.strptime(end_date_str,input_format)
    start_date = datetime.datetime.strptime(start_date_str,input_format)
    output = [d.strftime(output_format) for d in rrule(WEEKLY,dtstart=start_date,until=end_date)]

    return output

#sunday generator
def get_sunday_range(start_date_str,end_date_str,input_format="%Y%m%d",output_format="%Y%m%d"):
    """
    Return the list of sundays between start_date_str and end_date_str
    """
    end_date = datetime.datetime.strptime(end_date_str,input_format)
    start_date = datetime.datetime.strptime(start_date_str,input_format)
    #fast forward to sunday
    sunday_start = datetime.datetime.strptime((datetime.datetime.strftime(start_date,"%Y%W")+'Sunday'),"%Y%W%A")
    sunday_end = datetime.datetime.strptime((datetime.datetime.strftime(end_date,"%Y%W")+'Sunday'),"%Y%W%A")
    return [d.strftime(output_format) for d in rrule(WEEKLY,dtstart=sunday_start,until=sunday_end)]

def get_saturday_range(start_date_str,end_date_str,input_format="%Y%m%d",output_format="%Y%m%d"):
    """
    Return the list of sundays between start_date_str and end_date_str
    """
    end_date = datetime.datetime.strptime(end_date_str,input_format)
    start_date = datetime.datetime.strptime(start_date_str,input_format)
    #fast forward to saturday
    saturday_start = datetime.datetime.strptime((datetime.datetime.strftime(start_date,"%Y%W")+'Saturday'),"%Y%W%A")
    saturday_end = datetime.datetime.strptime((datetime.datetime.strftime(end_date,"%Y%W")+'Saturday'),"%Y%W%A")
    return [d.strftime(output_format) for d in rrule(WEEKLY,dtstart=sunday_start,until=sunday_end)]


#month generator
def get_month_range(start_month_str,end_month_str,input_format="%Y%m",output_format="%Y%m"):
    """
    Example: get_month_range("201603","201605")
    """
    start = datetime.datetime.strptime(start_month_str+"01",input_format+"%d")
    end = datetime.datetime.strptime(end_month_str+"01",input_format+"%d")
    output = [d.strftime(output_format) for d in rrule(MONTHLY,dtstart=start,until=end)]
    return output

#
def get_date_from_Ym(input_Ym, input_format="%Y%m",output_format="%Y%m%d"):
    """
    Example: get_date_from_Ym("201602")
    """
    last_day = get_last_day_of_month_from_Ym(input_Ym,0)
    first_day = input_Ym+"01"
    output = get_daterange(first_day,last_day)
    return output
#
def get_prev_ndays(upto_ymd, ndays_look_back):
    date = datetime.datetime.strptime(upto_ymd, '%Y%m%d')
    dates = []
    curr_date = date
    for i in range(ndays_look_back):
        dates.append(curr_date.strftime('%Y%m%d'))
        curr_date -= datetime.timedelta(1)
    dates = sorted(dates)
    return dates

def get_next_ndays(upto_ymd, ndays_look_forward):
    date = datetime.datetime.strptime(upto_ymd, '%Y%m%d')
    dates = []
    curr_date = date
    for i in range(ndays_look_forward):
        dates.append(curr_date.strftime('%Y%m%d'))
        curr_date += datetime.timedelta(1)
    dates = sorted(dates)
    return dates

def is_sunday(ymd,input_format="%Y%m%d"):
    return datetime.datetime.strptime(ymd,input_format).strftime("%A") == 'Sunday'

def is_last_date_of_month(ymd,input_format="%Y%m%d"):
    date = datetime.datetime.strptime(ymd,input_format)
    return date.day == calendar.monthrange(date.year, date.month)[1]


def get_hours_minutes_seconds(timedelta):
    '''
    Convert time delta to hours, minutes, seconds

    Parameters
    ----------
    timedelta : datetime.timedelta
        time delta between two time points. Ex: datetime.timedelta(0, 9, 494935)

    Returns
    -------
    three integer objects corresponding to number of hours, minutes and seconds
    '''
    total_seconds = timedelta.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds - (hours * 3600)) // 60
    seconds = total_seconds - (hours * 3600) - (minutes * 60)
    return hours, minutes, seconds


def get_last_dates(input_date,
                   periods=1,
                   input_format='%Y%m%d',
                   output_format='%Y%m%d'):
    '''
    Get list of last dates of months or weeks

    Parameters
    ----------
    input_date : str
        input date. Ex: '202106' or '20210630'
    input_format : str
        format of input time. Ex: '%Y%m' or '%Y%m%d'
    output_format : str
        format of output time. Ex: '%Y%m' or '%Y%m%d'
    periods : int
        number of time periods wanted to look back. Ex: 2

    Returns
    -------
    list of str
        list of last dates of months or weeks. Ex: ['20210630', '20210531']
    '''
    input_date = datetime.datetime.strptime(input_date, input_format)
    input_month = datetime.datetime.strftime(input_date, '%Y%m')

    months = get_prev_nmonths(input_month, nmonth_lookback=periods)
    dates = [get_last_day_of_month_from_Ym(month, output_format=output_format) \
        for month in months]
    return dates


def get_last_dates_range(start_date,
                         end_date,
                         input_format='%Y%m%d',
                         output_format='%Y%m%d'):
    '''
    Get list of last dates from starting date to ending date included.

    Parameters
    ----------
    start_date : str
        Starting date. Ex: '20200630'.
        Starting date is not neccessarily at the end of the month.
        I.e. '20200630' or 20200615 or '20200601' would result the same.
    end_date : str
        Ending date. Ex: '20210630'.
        Ending date is not neccessarily at the end of the month.
        I.e. '20210630' or 20210615 or '20210601' would result the same.
    input_format : str
        Format of input dates. Ex: '%Y%m%d'
    output_format : str
        Format of output dates. Ex: '%Y%m%d'

    Returns
    -------
    list of str
         List of last dates. Ex: ['20210630', '20210531', ...]
    '''
    start_date = datetime.datetime.strptime(start_date, input_format)
    end_date = datetime.datetime.strptime(end_date, input_format)

    start_month = datetime.datetime.strftime(start_date, '%Y%m')
    end_month = datetime.datetime.strftime(end_date, '%Y%m')

    months = get_month_range(start_month, end_month)
    dates = [get_last_day_of_month_from_Ym(month, output_format=output_format) \
        for month in months]
    return dates
